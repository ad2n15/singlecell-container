//! Implicitly scaled PCA: replaces Seurat's `ScaleData` + `RunPCA` without ever building
//! the dense scaled matrix (docs/RUST_ENGINE.md sections 1 and 8).
//!
//! With X the log-normalised cells x genes matrix (HVGs only), per-gene mean `mu` and
//! standard deviation `sigma`, the scaled matrix is `Z = (X - 1 mu^T) diag(1/sigma)`.
//! Randomised SVD only needs `Z v` and `Z^T u`, which are sparse products plus a rank-one
//! correction, so memory stays at the size of the sparse X.
//!
//! # Algorithm
//!
//! 1. One parallel pass over the CSR for per-gene mean and standard deviation (n - 1
//!    denominator, as `ScaleData`); constant genes get sd 1.
//! 2. Randomised SVD (Halko, Martinsson & Tropp 2011, algorithms 4.4 and 5.1) on the
//!    implicit operator: Gaussian test matrix (genes x l, l = n_comps + oversample) from
//!    ChaCha8 seeded with `seed`; range finder with `power_iters` subspace iterations,
//!    re-orthonormalising after every product; then the thin SVD of the small
//!    genes x l matrix `Z^T Q`.
//! 3. Embeddings `Q V_b diag(d)`, loadings `U_b`, all in f64, returned as f32.
//!
//! # Memory
//!
//! Besides the borrowed X: one cells x l f64 matrix (Q, overwritten in place; 320 bytes
//! per cell at l = 40, against 16 KB per cell for `ScaleData`'s dense 2,000-HVG matrix),
//! a few genes x l matrices, per-task chunk buffers, and the `Z^T U` accumulators (see
//! `operator.rs`; about threads x log2(blocks) genes x l matrices). No transposed copy of X.
//!
//! # Determinism
//!
//! Every reduction uses a partition fixed by the input shape and a fixed summation tree,
//! and every faer call runs with `Par::Seq` inside its task, so a given seed and input give
//! bitwise-identical output for any number of rayon threads (on the same machine and build).

use crate::device::Device;
use crate::error::{Error, Result};
use crate::types::{CsrView, Idx, Real, RowMajor};

#[cfg(feature = "cuda")]
mod cuda;
mod dense;
mod operator;

use operator::{Blocks, ScaledOp};

/// The implicit operator `Z` as the randomised SVD sees it: `Z v` and `Z^T u` on
/// column-major f64 faer matrices. Implemented by the CPU [`ScaledOp`] and, with feature
/// `cuda`, by [`cuda::CudaScaledOp`].
pub(crate) trait Operator {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    /// `out = Z v`, `v` genes x l, `out` cells x l.
    fn apply(&self, v: faer::MatRef<'_, f64>, out: faer::MatMut<'_, f64>) -> Result<()>;
    /// `Z^T u`, `u` cells x l; returns genes x l.
    fn apply_t(&self, u: faer::MatRef<'_, f64>) -> Result<faer::Mat<f64>>;
}

/// Parameters, with Seurat-compatible defaults.
#[derive(Clone, Debug)]
pub struct PcaParams {
    /// Number of components (Seurat `npcs`; pipeline default 30).
    pub n_comps: usize,
    /// Extra random directions for the range finder (Halko et al.). Default 30: scRNA-seq
    /// spectra flatten into a noise bulk after the cell-type components, and the extra
    /// directions are what lets PCs 10-30 converge (see `Default`).
    pub oversample: usize,
    /// Power iterations; more improves accuracy when the spectrum decays slowly. Default 10.
    pub power_iters: usize,
    /// RNG seed (Seurat `seed.use`; pipeline default 0). Same seed, same result.
    pub seed: u64,
    /// Divide genes by their standard deviation (Seurat `do.scale = TRUE`).
    pub scale: bool,
    /// Seurat's `scale.max` clip (default 10). The implicit operator cannot clip, so this is
    /// recorded for reference only and the differences are measured by the parity checks.
    pub clip: Option<f64>,
}

/// Defaults chosen on the parity runs (Seurat 5.0.1, top-10 PC subspace similarity against
/// unclipped `RunPCA`): `oversample = 10, power_iters = 4` gives 0.936 on the pbmc_like
/// fixture and 0.88-0.94 on 2k/5k synthetic PBMC-like data, because PCs past the cell-type
/// signal sit in a flat noise bulk; `oversample = 30, power_iters = 10` gives 0.9998,
/// 0.994 and 0.986 (signal PCs exact) for about 2.7x the time, still >25x faster than
/// `ScaleData` + `RunPCA`.
impl Default for PcaParams {
    fn default() -> Self {
        Self { n_comps: 30, oversample: 30, power_iters: 10, seed: 0, scale: true, clip: Some(10.0) }
    }
}

/// Output in Seurat's convention (`RunPCA` via irlba on the scaled matrix).
#[derive(Clone, Debug)]
pub struct PcaResult {
    /// Cell embeddings, cells x n_comps: U * diag(d) (Seurat `Embeddings`, Scanpy `X_pca`).
    pub embeddings: RowMajor,
    /// Gene loadings, genes x n_comps: V (Seurat `Loadings`, Scanpy `varm["PCs"]`).
    pub loadings: RowMajor,
    /// Standard deviations of the components: d / sqrt(n_cells - 1) (Seurat `Stdev`).
    pub stdev: Vec<f32>,
    /// Per-gene means and standard deviations used for the implicit scaling.
    pub gene_mean: Vec<f32>,
    pub gene_sd: Vec<f32>,
}

/// PCA of the implicitly centred (and, if `params.scale`, scaled) matrix `x`.
///
/// - `x`: cells x genes, log-normalised, typically restricted to the HVGs.
/// - Genes with zero variance get sd 1 (they contribute nothing), as in Seurat.
/// - Components are ordered by decreasing singular value; the sign of each component is
///   fixed deterministically (largest-magnitude loading positive) so results are stable.
/// - Parallel over rows/columns with rayon; deterministic for a fixed seed regardless of
///   the number of threads.
///
/// Details:
/// - `gene_sd` is the sd actually used: the sample sd (n - 1 denominator) for scaled genes,
///   1 for constant genes, and 1 for every gene when `params.scale` is false.
/// - `l = n_comps + oversample` is capped at `min(n_cells, n_genes)`.
/// - `params.clip` is validated (must be positive, `inf` allowed) but not applied.
/// - Errors: `InvalidParameter` for `n_comps == 0`, `n_comps >= min(n_cells, n_genes)` or a
///   non-positive / NaN `clip`; `InvalidInput` for malformed CSR structure (including
///   duplicate column indices within a row) or non-finite values.
pub fn implicit_scaled_pca<I: Idx, T: Real>(x: CsrView<'_, I, T>, params: &PcaParams) -> Result<PcaResult> {
    implicit_scaled_pca_on(x, params, Device::Cpu)
}

/// [`implicit_scaled_pca`] on a chosen [`Device`] (docs/scfast-integration.md Phase 6).
///
/// `Device::Cpu` is exactly [`implicit_scaled_pca`]. `Device::Cuda(i)` runs the two sparse
/// products of every randomised-SVD step on the GPU (NVRTC CSR kernels with the mean / 1/sd
/// correction applied on device, f32) and keeps the gene statistics, TSQR and the small SVD
/// on the CPU in f64; it needs the `cuda` feature, a visible device and enough device memory
/// for the CSR plus a few cells x l buffers, otherwise it returns [`Error::Gpu`] without
/// touching the CPU path (callers that asked for `"auto"` can retry on the CPU). The GPU
/// result is not bitwise equal to the CPU one (f32 products, atomics); the parity target
/// is a top-10 PC subspace similarity above 0.99.
pub fn implicit_scaled_pca_on<I: Idx, T: Real>(x: CsrView<'_, I, T>, params: &PcaParams, device: Device) -> Result<PcaResult> {
    validate(&x, params)?;
    let (n, p, k) = (x.n_rows, x.n_cols, params.n_comps);
    let blocks = Blocks::new(n, p, x.data.len());

    let stats = operator::gene_stats(&x, blocks)?;
    let gene_sd: Vec<f64> = (0..p)
        .map(|j| if !params.scale || stats.constant[j] || stats.sd[j] == 0.0 { 1.0 } else { stats.sd[j] })
        .collect();
    let inv_sd: Vec<f64> = (0..p).map(|j| if stats.constant[j] { 0.0 } else { 1.0 / gene_sd[j] }).collect();

    let l = k.saturating_add(params.oversample).min(n.min(p));
    let (q, ub, d, vb) = match device {
        Device::Cpu => {
            let op = ScaledOp::new(x, &stats.mean, &inv_sd, blocks);
            randomized_svd(&op, l, params.power_iters, params.seed)?
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(ordinal) => {
            let op = cuda::CudaScaledOp::new(x, &stats.mean, &inv_sd, l, ordinal)?;
            if p <= GRAM_MAX_GENES {
                gram_svd(&op, l, params.power_iters, params.seed)?
            } else {
                randomized_svd(&op, l, params.power_iters, params.seed)?
            }
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => {
            return Err(Error::gpu("scfast was built without CUDA support (feature `cuda`); use Device::Cpu"));
        }
    };

    // Sign convention: the largest-magnitude loading of each component is positive (ties go
    // to the lowest gene index).
    let signs: Vec<f64> = (0..k)
        .map(|c| {
            let col = ub.col(c);
            let mut best = 0usize;
            for j in 1..p {
                if col[j].abs() > col[best].abs() {
                    best = j;
                }
            }
            if col[best] < 0.0 { -1.0 } else { 1.0 }
        })
        .collect();

    // Constant genes have an exactly zero row in Z^T Q; the dense SVD leaves ~1e-24 round-off
    // there, so their loadings are set to the exact zero.
    let mut loadings = RowMajor::zeros(p, k);
    for j in (0..p).filter(|&j| !stats.constant[j]) {
        for c in 0..k {
            loadings.data[j * k + c] = (ub[(j, c)] * signs[c]) as f32;
        }
    }
    let embeddings = embed(q.as_ref(), vb.as_ref(), &d, &signs, k);
    let denom = ((n - 1) as f64).sqrt();
    Ok(PcaResult {
        embeddings,
        loadings,
        stdev: d[..k].iter().map(|s| (s / denom) as f32).collect(),
        gene_mean: stats.mean.iter().map(|&m| m as f32).collect(),
        gene_sd: gene_sd.iter().map(|&s| s as f32).collect(),
    })
}

fn validate<I: Idx, T: Real>(x: &CsrView<'_, I, T>, params: &PcaParams) -> Result<()> {
    let (n, p) = (x.n_rows, x.n_cols);
    // Cheap structural checks; per-entry checks happen in the statistics pass.
    if x.indptr.len() != n + 1 {
        return Err(Error::InvalidInput(format!("indptr has {} entries, expected n_rows + 1 = {}", x.indptr.len(), n + 1)));
    }
    if x.indices.len() != x.data.len() {
        return Err(Error::InvalidInput(format!(
            "indices ({}) and data ({}) differ in length",
            x.indices.len(),
            x.data.len()
        )));
    }
    if x.indptr[0].to_usize() != 0 || x.indptr[n].to_usize() != x.data.len() {
        return Err(Error::InvalidInput("indptr must start at 0 and end at nnz".into()));
    }
    if params.n_comps == 0 {
        return Err(Error::InvalidParameter("n_comps must be at least 1".into()));
    }
    if params.n_comps >= n.min(p) {
        return Err(Error::InvalidParameter(format!(
            "n_comps ({}) must be smaller than both the number of cells ({n}) and of genes ({p})",
            params.n_comps
        )));
    }
    if let Some(c) = params.clip {
        if c.is_nan() || c <= 0.0 {
            return Err(Error::InvalidParameter(format!("clip must be positive (or None), got {c}")));
        }
    }
    Ok(())
}

/// Randomised SVD of the implicit operator. Returns (Q: n x l orthonormal, U_b: p x l,
/// d: l singular values, V_b: l x l) with `Z ≈ (Q V_b) diag(d) U_b^T`.
fn randomized_svd<O: Operator>(
    op: &O,
    l: usize,
    power_iters: usize,
    seed: u64,
) -> Result<(faer::Mat<f64>, faer::Mat<f64>, Vec<f64>, faer::Mat<f64>)> {
    let (n, p) = (op.nrows(), op.ncols());
    let omega = dense::gaussian(p, l, seed);
    let mut q = faer::Mat::<f64>::zeros(n, l);
    op.apply(omega.as_ref(), q.as_mut())?;
    drop(omega);
    dense::orthonormalize(q.as_mut());
    for _ in 0..power_iters {
        let mut w = op.apply_t(q.as_ref())?;
        dense::orthonormalize(w.as_mut());
        op.apply(w.as_ref(), q.as_mut())?;
        dense::orthonormalize(q.as_mut());
    }
    // B^T = Z^T Q (p x l) = U_b diag(d) V_b^T, so Z ≈ Q B = (Q V_b) diag(d) U_b^T.
    let bt = op.apply_t(q.as_ref())?;
    let (ub, d, vb) = dense::thin_svd(bt.as_ref())?;
    Ok((q, ub, d, vb))
}

/// Embeddings `Q V_b[:, :k] diag(d_k signs)` as row-major f32, parallel over fixed row
/// chunks (each a sequential faer product, so thread-count independent).
/// Largest gene count for which the CUDA path forms the Gram matrix (8 p^2 bytes on the
/// device; 0.5 GiB at this size).
#[cfg(feature = "cuda")]
const GRAM_MAX_GENES: usize = 8192;

/// [`randomized_svd`] for the device: Z^T Z (genes x genes) in one pass over the data,
/// subspace iteration on it on the host (each step equals one `apply_t` + `apply`, so the
/// same `power_iters` gives the same accuracy), Rayleigh-Ritz on the l x l projection, and
/// one `apply` for the cell scores. Avoids 2 (power_iters + 1) cells x l round trips and
/// their host QR factorisations. Returns (U, V, S, I) so that `embed` gives Z V.
#[cfg(feature = "cuda")]
fn gram_svd(
    op: &cuda::CudaScaledOp,
    l: usize,
    power_iters: usize,
    seed: u64,
) -> Result<(faer::Mat<f64>, faer::Mat<f64>, Vec<f64>, faer::Mat<f64>)> {
    let g = op.gram()?;
    let (n, p) = (op.nrows(), op.ncols());
    let mut v = dense::gaussian(p, l, seed);
    for _ in 0..=power_iters {
        dense::orthonormalize(v.as_mut());
        v = &g * &v;
    }
    dense::orthonormalize(v.as_mut());
    let h = v.transpose() * &g * &v;
    let (w, lambda, _) = dense::thin_svd(h.as_ref())?;
    let vr = &v * &w;
    let s: Vec<f64> = lambda.iter().map(|&x| x.max(0.0).sqrt()).collect();
    let mut u = faer::Mat::<f64>::zeros(n, l);
    op.apply(vr.as_ref(), u.as_mut())?;
    for (c, &sc) in s.iter().enumerate() {
        let inv = if sc > 0.0 { 1.0 / sc } else { 0.0 };
        u.col_mut(c).iter_mut().for_each(|y| *y *= inv);
    }
    Ok((u, vr, s, faer::Mat::<f64>::identity(l, l)))
}

fn embed(q: faer::MatRef<'_, f64>, vb: faer::MatRef<'_, f64>, d: &[f64], signs: &[f64], k: usize) -> RowMajor {
    use rayon::prelude::*;
    const ROWS: usize = 1024;
    let n = q.nrows();
    let w = faer::Mat::<f64>::from_fn(vb.nrows(), k, |t, c| vb[(t, c)] * d[c] * signs[c]);
    let mut out = RowMajor::zeros(n, k);
    out.data.par_chunks_mut(ROWS * k).zip(q.par_row_chunks(ROWS)).for_each(|(dst, qc)| {
        let mut e = faer::Mat::<f64>::zeros(qc.nrows(), k);
        faer::linalg::matmul::matmul(e.as_mut(), faer::Accum::Replace, qc, w.as_ref(), 1.0, faer::Par::Seq);
        for r in 0..qc.nrows() {
            for c in 0..k {
                dst[r * k + c] = e[(r, c)] as f32;
            }
        }
    });
    out
}
