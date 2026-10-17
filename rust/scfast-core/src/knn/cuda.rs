//! kNN on a CUDA device (feature `cuda`): the fused `knn` kernel (exact f32 distances and
//! per-query top-k in one pass, see `cuda/mod.rs`), with the same exact f64 re-ranking and
//! output contract as the CPU methods.
//!
//! Data are centred on the host (f64 mean, rounded to f32) and uploaded once. Queries go in
//! blocks of `qb` (bounded by the candidate buffers' memory); per block the kernel keeps the
//! `cap = k - 1 + slack` smallest f32 squared distances per query, and the host re-ranks
//! those candidates by exact f64 distances (`exact_d2`, as `exact.rs`) and writes the row.
//!
//! The error-bound certificate of the CPU exact search is not applied: the result is exact
//! whenever the true k - 1 neighbours are among the `cap` best f32 distances, which holds
//! for all but pathological ties; the parity target for the GPU path is a neighbour
//! overlap of at least 90% with the CPU result. Values so large that f32 squared distances
//! could overflow are handed to the CPU exact search.

use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use rayon::prelude::*;

use super::{cmp_neighbour, exact_d2, write_row};
use crate::cuda::{drv_err, guard, Gpu, BLOCK, KNN_CAP_MAX};
use crate::error::{Error, Result};
use crate::types::{Knn, RowMajor};

/// Queries per kernel launch, at most (bounds the candidate buffers: 8 * cap bytes each).
const MAX_QB: usize = 1 << 20;
/// Same overflow guard as `exact.rs`.
const F32_SAFE_SQ_NORM: f64 = 1e36;

pub(super) fn cuda_knn(x: &RowMajor, k: usize, ordinal: usize) -> Result<Knn> {
    let (n, d) = (x.n_rows, x.n_cols);
    let kk = k - 1;
    if n > i32::MAX as usize || d > i32::MAX as usize {
        return Err(Error::gpu(format!("{n} cells x {d} dims exceeds the CUDA kernel index range")));
    }
    if kk > KNN_CAP_MAX {
        return Err(Error::gpu(format!("k = {k} is above the CUDA kNN limit of {} (k counts the cell itself); use the CPU", KNN_CAP_MAX + 1)));
    }
    let slack = (kk / 2).max(8);
    let cap = (kk + slack).min(n - 1).min(KNN_CAP_MAX).max(kk);

    // Centre (f64 mean, rounded to f32), as the CPU exact search does.
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
    let max_norm = xc
        .par_chunks_mut(d)
        .enumerate()
        .map(|(i, row)| {
            let mut s = 0f64;
            for (t, (y, &v)) in row.iter_mut().zip(x.row(i)).enumerate() {
                *y = (v as f64 - mean[t]) as f32;
                s += *y as f64 * *y as f64;
            }
            s
        })
        .reduce(|| 0.0, f64::max);
    if max_norm * 4.0 > F32_SAFE_SQ_NORM {
        // f32 squared distances could overflow: the CPU exact search handles this case.
        return Ok(super::exact::exact_knn(x, k));
    }

    let gpu = Gpu::get(ordinal)?;
    let base = 4 * n * d;
    gpu.ensure_fits(base + 8 * cap, &format!("kNN on {n} cells x {d} dims"))?;
    let free = gpu.free_bytes()?;
    let qb = (free.saturating_sub(base) / 2 / (8 * cap)).min(MAX_QB).min(n);
    if qb == 0 {
        return Err(Error::gpu(format!("out of device memory: kNN on {n} cells, {:.2} GiB free", free as f64 / (1u64 << 30) as f64)));
    }

    let mut indices = vec![0u32; n * k];
    let mut distances = vec![0f32; n * k];
    run_blocks(&gpu, x, &xc, n, d, kk, cap, qb, &mut indices, &mut distances)?;
    Ok(Knn { n, k, indices, distances })
}

#[allow(clippy::too_many_arguments)]
fn run_blocks(
    gpu: &Arc<Gpu>,
    x: &RowMajor,
    xc: &[f32],
    n: usize,
    d: usize,
    kk: usize,
    cap: usize,
    qb: usize,
    indices: &mut [u32],
    distances: &mut [f32],
) -> Result<()> {
    let k = kk + 1;
    let (kernel, tp) = gpu.knn_kernel(d, cap)?;
    guard("CUDA kNN", || {
        let xc_d = gpu.upload(xc, "upload embeddings")?;
        let mut out_idx = gpu.alloc_zeros::<u32>(qb * cap, "alloc candidate indices")?;
        let mut out_d2 = gpu.alloc_zeros::<f32>(qb * cap, "alloc candidate distances")?;
        let mut host_idx = vec![0u32; qb * cap];

        let mut q0 = 0usize;
        while q0 < n {
            let nq = qb.min(n - q0);
            let (ni, q0i, nqi) = (n as i32, q0 as i32, nq as i32);
            let cfg = LaunchConfig {
                grid_dim: (nq.div_ceil(BLOCK as usize) as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: (4 * tp * d) as u32,
            };
            // SAFETY: argument order and types match the `knn` kernel compiled for this
            // (d, cap); the shared tile holds tp * d floats.
            unsafe {
                gpu.stream
                    .launch_builder(&kernel)
                    .arg(&xc_d)
                    .arg(&ni)
                    .arg(&q0i)
                    .arg(&nqi)
                    .arg(&mut out_idx)
                    .arg(&mut out_d2)
                    .launch(cfg)
            }
            .map_err(|e| drv_err("launch knn", e))?;
            gpu.stream.memcpy_dtoh(&out_idx, &mut host_idx).map_err(|e| drv_err("download candidates", e))?;
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
