//! kNN on a CUDA device (feature `cuda`): cuBLAS GEMM for the cross terms and the `topk`
//! kernel for candidate selection, with the same exact f64 re-ranking and output contract
//! as the CPU methods.
//!
//! Data are centred on the host (f64 mean, rounded to f32) and uploaded once with their
//! squared norms. For each block of `qb` queries one SGEMM gives the dot products of the
//! block against every point (an n x qb column-major buffer, bounded by free memory), the
//! `topk` kernel keeps the `cap = k - 1 + slack` smallest estimates `|r|^2 - 2 q.r` per
//! query (the per-query `|q|^2` is a constant), and the host re-ranks those candidates by
//! exact f64 distances (`exact_d2`, as `exact.rs`) and writes the row.
//!
//! The error-bound certificate of the CPU exact search is not applied: the result is exact
//! whenever the true k - 1 neighbours are among the `cap` best f32 estimates, which holds
//! for all but pathological ties; the parity target for the GPU path is a neighbour
//! overlap of at least 90% with the CPU result. Values so large that the f32 cross terms
//! could overflow are handed to the CPU exact search.

use std::sync::Arc;

use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::PushKernelArg;
use rayon::prelude::*;

use super::{cmp_neighbour, exact_d2, write_row};
use crate::cuda::{blas_err, drv_err, guard, Gpu, BLOCK, TOPK_CAP_MAX};
use crate::error::{Error, Result};
use crate::types::{Knn, RowMajor};

/// Upper bound on the GEMM score buffer (bytes); also capped at half the free memory.
const SCORE_BUDGET: usize = 512 << 20;
/// Queries per GEMM block, at most.
const MAX_QB: usize = 4096;
/// Same overflow guard as `exact.rs`.
const F32_SAFE_SQ_NORM: f64 = 1e36;

pub(super) fn cuda_knn(x: &RowMajor, k: usize, ordinal: usize) -> Result<Knn> {
    let (n, d) = (x.n_rows, x.n_cols);
    let kk = k - 1;
    if n > i32::MAX as usize || d > i32::MAX as usize {
        return Err(Error::gpu(format!("{n} cells x {d} dims exceeds the CUDA kernel index range")));
    }
    if kk > TOPK_CAP_MAX {
        return Err(Error::gpu(format!("k = {k} is above the CUDA kNN limit of {} (k counts the cell itself); use the CPU", TOPK_CAP_MAX + 1)));
    }
    let slack = (kk / 2).max(8);
    let cap = (kk + slack).min(n - 1).min(TOPK_CAP_MAX).max(kk);

    // Centre (f64 mean, rounded to f32) and square norms, as the CPU exact search does.
    let mut mean = vec![0f64; d];
    for i in 0..n {
        for (m, &v) in mean.iter_mut().zip(x.row(i)) {
            *m += v as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    let mut xc = vec![0f32; n * d];
    let mut norms = vec![0f32; n];
    let mut max_norm = 0f64;
    xc.par_chunks_mut(d).zip(norms.par_iter_mut()).enumerate().for_each(|(i, (row, nrm))| {
        let xi = x.row(i);
        let mut s = 0f64;
        for t in 0..d {
            let y = (xi[t] as f64 - mean[t]) as f32;
            row[t] = y;
            s += y as f64 * y as f64;
        }
        *nrm = s as f32;
    });
    for &v in &norms {
        max_norm = max_norm.max(v as f64);
    }
    if max_norm * 4.0 > F32_SAFE_SQ_NORM {
        // f32 cross terms could overflow: the CPU exact search handles this case.
        return Ok(super::exact::exact_knn(x, k));
    }

    let gpu = Gpu::get(ordinal)?;
    let base = 4 * (n * d + n);
    gpu.ensure_fits(base + 4 * n, &format!("kNN on {n} cells x {d} dims"))?;
    let free = gpu.free_bytes()?;
    let budget = SCORE_BUDGET.min(free.saturating_sub(base) / 2);
    let qb = (budget / (4 * n)).min(MAX_QB).min(n);
    if qb == 0 {
        return Err(Error::gpu(format!(
            "out of device memory: kNN on {n} cells needs at least {:.2} GiB for one query block, {:.2} GiB free",
            (base + 4 * n) as f64 / (1u64 << 30) as f64,
            free as f64 / (1u64 << 30) as f64
        )));
    }

    let mut indices = vec![0u32; n * k];
    let mut distances = vec![0f32; n * k];
    run_blocks(&gpu, x, &xc, &norms, n, d, kk, cap, qb, &mut indices, &mut distances)?;
    Ok(Knn { n, k, indices, distances })
}

#[allow(clippy::too_many_arguments)]
fn run_blocks(
    gpu: &Arc<Gpu>,
    x: &RowMajor,
    xc: &[f32],
    norms: &[f32],
    n: usize,
    d: usize,
    kk: usize,
    cap: usize,
    qb: usize,
    indices: &mut [u32],
    distances: &mut [f32],
) -> Result<()> {
    let k = kk + 1;
    guard("CUDA kNN", || {
        let xc_d = gpu.upload(xc, "upload embeddings")?;
        let norms_d = gpu.upload(norms, "upload norms")?;
        let mut scores = gpu.alloc_zeros::<f32>(n * qb, "alloc score block")?;
        let mut out_idx = gpu.alloc_zeros::<u32>(qb * cap, "alloc candidate indices")?;
        let mut out_est = gpu.alloc_zeros::<f32>(qb * cap, "alloc candidate estimates")?;
        let mut host_idx = vec![0u32; qb * cap];
        let mut host_est = vec![0f32; qb * cap];

        let mut q0 = 0usize;
        while q0 < n {
            let nq = qb.min(n - q0);
            // scores (n x nq, column-major) = Xc (n x d) . Xc[q0..q0+nq]^T: with Xc stored
            // row-major it is the column-major d x n matrix A, so C = A^T B with
            // B = A[:, q0..q0+nq] (ld = d).
            let cfg = GemmConfig {
                transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: nq as i32,
                k: d as i32,
                alpha: 1.0f32,
                lda: d as i32,
                ldb: d as i32,
                beta: 0.0f32,
                ldc: n as i32,
            };
            let b = xc_d.slice(q0 * d..(q0 + nq) * d);
            // SAFETY: dimensions and leading dimensions match the buffers above.
            unsafe { gpu.blas.gemm(cfg, &xc_d, &b, &mut scores) }.map_err(|e| blas_err("cublasSgemm", e))?;
            let (ni, q0i, capi) = (n as i32, q0 as i32, cap as i32);
            // SAFETY: argument order and types match the `topk` kernel; one block per
            // query, BLOCK threads as the kernel's shared arrays assume.
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.topk)
                    .arg(&scores)
                    .arg(&norms_d)
                    .arg(&ni)
                    .arg(&q0i)
                    .arg(&capi)
                    .arg(&mut out_idx)
                    .arg(&mut out_est)
                    .launch(cudarc::driver::LaunchConfig { grid_dim: (nq as u32, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 })
            }
            .map_err(|e| drv_err("launch topk", e))?;
            gpu.stream.memcpy_dtoh(&out_idx, &mut host_idx).map_err(|e| drv_err("download candidates", e))?;
            gpu.stream.memcpy_dtoh(&out_est, &mut host_est).map_err(|e| drv_err("download candidates", e))?;
            gpu.sync("download candidates")?;

            // Exact re-ranking of the candidates on the host.
            let idx_blk = &mut indices[q0 * k..(q0 + nq) * k];
            let dist_blk = &mut distances[q0 * k..(q0 + nq) * k];
            let fallback: usize = idx_blk
                .par_chunks_mut(k)
                .zip(dist_blk.par_chunks_mut(k))
                .enumerate()
                .map_init(
                    || Vec::with_capacity(cap),
                    |cand, (t, (irow, drow))| {
                        let i = q0 + t;
                        let xi = x.row(i);
                        cand.clear();
                        for &j in &host_idx[t * cap..(t + 1) * cap] {
                            if j != u32::MAX && j as usize != i {
                                cand.push((exact_d2(xi, x.row(j as usize)), j));
                            }
                        }
                        let mut fb = 0;
                        if cand.len() < kk {
                            // Cannot happen when cap <= n - 1; kept as a safety net.
                            super::brute_one(x, i, kk, cand);
                            fb = 1;
                        }
                        cand.sort_unstable_by(cmp_neighbour);
                        write_row(i, cand, irow, drow);
                        fb
                    },
                )
                .sum();
            debug_assert_eq!(fallback, 0);
            q0 += nq;
        }
        Ok(())
    })
}
