//! The implicitly scaled operator on a CUDA device (feature `cuda`).
//!
//! The CSR of the HVG matrix is uploaded once (i64 row offsets, u32 column indices, f32
//! values) with the per-gene mean and 1/sd. Each Krylov step then does:
//!
//! - `apply`:   host builds `vs = diag(1/sd) V` (row-major f32) and `shift = mu^T vs` (f64,
//!   rounded once), uploads them, runs the `zv` kernel, downloads the cells x l result.
//! - `apply_t`: host uploads `U` (f32, column-major) and its column sums (f64 on the host),
//!   zeroes the genes x l accumulator, runs `ztu` (atomic scatter) and `ztu_finish` (mean
//!   and 1/sd correction), downloads the genes x l result.
//!
//! All device buffers are allocated at construction for the fixed `l`, so the working set
//! is known up front and checked against free memory; an allocation that still fails is
//! reported as `Error::Gpu("out of device memory ...")`.

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, DeviceRepr, PushKernelArg};
use faer::reborrow::ReborrowMut;
use faer::{Mat, MatMut, MatRef};

use crate::cuda::{drv_err, guard, Gpu};
use crate::error::{Error, Result};
use crate::types::{CsrView, Idx, Real};

/// Elements per host-to-device staging chunk for the CSR upload (16 MiB of f32).
const STAGE: usize = 4 << 20;

pub(super) struct CudaScaledOp {
    gpu: Arc<Gpu>,
    n: usize,
    p: usize,
    l: usize,
    indptr: CudaSlice<i64>,
    indices: CudaSlice<u32>,
    vals: CudaSlice<f32>,
    mean_d: CudaSlice<f32>,
    inv_sd_d: CudaSlice<f32>,
    mean: Vec<f64>,
    inv_sd: Vec<f64>,
    // Scratch, reused by every product (apply takes &self, hence RefCell).
    vs: RefCell<CudaSlice<f32>>,
    shift: RefCell<CudaSlice<f32>>,
    colsum: RefCell<CudaSlice<f32>>,
    dense_n: RefCell<CudaSlice<f32>>,
    acc: RefCell<CudaSlice<f32>>,
    out_p: RefCell<CudaSlice<f32>>,
    host_n: RefCell<Vec<f32>>,
    host_p: RefCell<Vec<f32>>,
}

impl CudaScaledOp {
    pub fn new<I: Idx, T: Real>(x: CsrView<'_, I, T>, mean: &[f64], inv_sd: &[f64], l: usize, ordinal: usize) -> Result<Self> {
        let (n, p, nnz) = (x.n_rows, x.n_cols, x.data.len());
        if n > i32::MAX as usize || p > i32::MAX as usize || l > i32::MAX as usize {
            return Err(Error::gpu(format!("{n} cells x {p} genes exceeds the CUDA kernel index range")));
        }
        let gpu = Gpu::get(ordinal)?;
        let needed = 8 * (n + 1) + 8 * nnz + 8 * p + 4 * (3 * p * l + n * l + 2 * l);
        gpu.ensure_fits(needed, &format!("PCA on {n} cells x {p} genes ({nnz} non-zeros, l = {l})"))?;
        guard("CUDA PCA setup", || {
            let indptr = upload_chunked(&gpu, n + 1, |r| x.indptr[r.clone()].iter().map(|v| v.to_usize() as i64).collect(), "upload indptr")?;
            let indices = upload_chunked(&gpu, nnz, |r| x.indices[r.clone()].iter().map(|v| v.to_usize() as u32).collect(), "upload indices")?;
            let vals = upload_chunked(&gpu, nnz, |r| x.data[r.clone()].iter().map(|v| v.to_f32()).collect(), "upload values")?;
            let mean_d = gpu.upload(&mean.iter().map(|&m| m as f32).collect::<Vec<f32>>(), "upload gene means")?;
            let inv_sd_d = gpu.upload(&inv_sd.iter().map(|&s| s as f32).collect::<Vec<f32>>(), "upload gene 1/sd")?;
            let vs = gpu.alloc_zeros::<f32>(p * l, "alloc vs")?;
            let shift = gpu.alloc_zeros::<f32>(l, "alloc shift")?;
            let colsum = gpu.alloc_zeros::<f32>(l, "alloc colsum")?;
            let dense_n = gpu.alloc_zeros::<f32>(n * l, "alloc cells x l")?;
            let acc = gpu.alloc_zeros::<f32>(p * l, "alloc accumulator")?;
            let out_p = gpu.alloc_zeros::<f32>(p * l, "alloc genes x l")?;
            Ok(Self {
                gpu: gpu.clone(),
                n,
                p,
                l,
                indptr,
                indices,
                vals,
                mean_d,
                inv_sd_d,
                mean: mean.to_vec(),
                inv_sd: inv_sd.to_vec(),
                vs: RefCell::new(vs),
                shift: RefCell::new(shift),
                colsum: RefCell::new(colsum),
                dense_n: RefCell::new(dense_n),
                acc: RefCell::new(acc),
                out_p: RefCell::new(out_p),
                host_n: RefCell::new(vec![0.0; n * l]),
                host_p: RefCell::new(vec![0.0; p * l]),
            })
        })
    }
}

/// Allocate `len` elements on the device and fill them through a bounded host staging
/// buffer, so the host never holds a full converted copy of the CSR.
fn upload_chunked<T: DeviceRepr>(gpu: &Gpu, len: usize, chunk: impl Fn(&std::ops::Range<usize>) -> Vec<T>, step: &str) -> Result<CudaSlice<T>> {
    // SAFETY: every element is written below before the slice is read.
    let mut dev = unsafe { gpu.stream.alloc::<T>(len) }.map_err(|e| drv_err(step, e))?;
    let mut start = 0;
    while start < len {
        let end = (start + STAGE).min(len);
        let host = chunk(&(start..end));
        let mut view = dev.slice_mut(start..end);
        gpu.stream.memcpy_htod(&host, &mut view).map_err(|e| drv_err(step, e))?;
        start = end;
    }
    gpu.sync(step)?;
    Ok(dev)
}

impl super::Operator for CudaScaledOp {
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.p
    }

    fn apply(&self, v: MatRef<'_, f64>, mut out: MatMut<'_, f64>) -> Result<()> {
        let (n, p, l) = (self.n, self.p, self.l);
        assert!(v.nrows() == p && v.ncols() == l && out.nrows() == n && out.ncols() == l);
        // vs = diag(inv_sd) v (row-major) and shift = mu^T vs, both in f64 first.
        let mut vs64 = vec![0.0f64; p * l];
        for c in 0..l {
            let col = v.col(c);
            for j in 0..p {
                vs64[j * l + c] = col[j] * self.inv_sd[j];
            }
        }
        let mut shift64 = vec![0.0f64; l];
        for j in 0..p {
            let m = self.mean[j];
            if m != 0.0 {
                for (s, w) in shift64.iter_mut().zip(&vs64[j * l..(j + 1) * l]) {
                    *s += m * w;
                }
            }
        }
        let mut host_p = self.host_p.borrow_mut();
        for (h, &w) in host_p.iter_mut().zip(&vs64) {
            *h = w as f32;
        }
        let shift32: Vec<f32> = shift64.iter().map(|&s| s as f32).collect();
        let mut vs = self.vs.borrow_mut();
        let mut shift = self.shift.borrow_mut();
        let mut dense = self.dense_n.borrow_mut();
        let mut host_n = self.host_n.borrow_mut();
        guard("CUDA PCA Z v", || {
            self.gpu.upload_into(&host_p, &mut vs, "upload vs")?;
            self.gpu.upload_into(&shift32, &mut shift, "upload shift")?;
            let (ni, li) = (n as i32, l as i32);
            // SAFETY: argument types and order match the `zv` kernel signature; every buffer
            // has the length the kernel indexes.
            unsafe {
                self.gpu
                    .stream
                    .launch_builder(&self.gpu.zv)
                    .arg(&self.indptr)
                    .arg(&self.indices)
                    .arg(&self.vals)
                    .arg(&*vs)
                    .arg(&*shift)
                    .arg(&mut *dense)
                    .arg(&ni)
                    .arg(&li)
                    .launch(Gpu::cfg_1d(n * l))
            }
            .map_err(|e| drv_err("launch zv", e))?;
            self.gpu.download_into(&dense, &mut host_n, "download Z v")
        })?;
        for c in 0..l {
            let src = &host_n[c * n..(c + 1) * n];
            let mut col = out.rb_mut().col_mut(c);
            for (r, &y) in src.iter().enumerate() {
                col[r] = y as f64;
            }
        }
        Ok(())
    }

    fn apply_t(&self, u: MatRef<'_, f64>) -> Result<Mat<f64>> {
        let (n, p, l) = (self.n, self.p, self.l);
        assert!(u.nrows() == n && u.ncols() == l);
        let mut host_n = self.host_n.borrow_mut();
        let mut colsum64 = vec![0.0f64; l];
        for c in 0..l {
            let col = u.col(c);
            let dst = &mut host_n[c * n..(c + 1) * n];
            let mut s = 0.0f64;
            for (r, y) in dst.iter_mut().enumerate() {
                let v = col[r];
                s += v;
                *y = v as f32;
            }
            colsum64[c] = s;
        }
        let colsum32: Vec<f32> = colsum64.iter().map(|&s| s as f32).collect();
        let mut dense = self.dense_n.borrow_mut();
        let mut colsum = self.colsum.borrow_mut();
        let mut acc = self.acc.borrow_mut();
        let mut out_p = self.out_p.borrow_mut();
        let mut host_p = self.host_p.borrow_mut();
        guard("CUDA PCA Z^T u", || {
            self.gpu.upload_into(&host_n, &mut dense, "upload u")?;
            self.gpu.upload_into(&colsum32, &mut colsum, "upload column sums")?;
            self.gpu.stream.memset_zeros(&mut *acc).map_err(|e| drv_err("zero accumulator", e))?;
            let (ni, pi, li) = (n as i32, p as i32, l as i32);
            // SAFETY: as in `apply`; the `ztu` and `ztu_finish` signatures are matched.
            unsafe {
                self.gpu
                    .stream
                    .launch_builder(&self.gpu.ztu)
                    .arg(&self.indptr)
                    .arg(&self.indices)
                    .arg(&self.vals)
                    .arg(&*dense)
                    .arg(&mut *acc)
                    .arg(&ni)
                    .arg(&li)
                    .launch(Gpu::cfg_1d(n * l))
            }
            .map_err(|e| drv_err("launch ztu", e))?;
            unsafe {
                self.gpu
                    .stream
                    .launch_builder(&self.gpu.ztu_finish)
                    .arg(&*acc)
                    .arg(&self.mean_d)
                    .arg(&self.inv_sd_d)
                    .arg(&*colsum)
                    .arg(&mut *out_p)
                    .arg(&pi)
                    .arg(&li)
                    .launch(Gpu::cfg_1d(p * l))
            }
            .map_err(|e| drv_err("launch ztu_finish", e))?;
            self.gpu.download_into(&out_p, &mut host_p, "download Z^T u")
        })?;
        Ok(Mat::from_fn(p, l, |j, c| host_p[j * l + c] as f64))
    }
}
