//! Implicitly scaled PCA: replaces Seurat's `ScaleData` + `RunPCA` without ever building
//! the dense scaled matrix (docs/RUST_ENGINE.md sections 1 and 8).
//!
//! With X the log-normalised cells x genes matrix (HVGs only), per-gene mean `mu` and
//! standard deviation `sigma`, the scaled matrix is `Z = (X - 1 mu^T) diag(1/sigma)`.
//! The SVD only needs `Z v` and `Z^T u`, which are sparse products plus a rank-one
//! correction (and a small sparse correction for Seurat's clip), so memory stays at the
//! size of the sparse X.
//!
//! # Algorithm
//!
//! 1. One parallel pass over the CSR for per-gene mean and standard deviation (n - 1
//!    denominator, as `ScaleData`); constant genes get sd 1.
//! 2. Block Krylov SVD (Musco & Musco 2015) on the implicit operator: Gaussian test matrix
//!    (genes x b, b = n_comps + oversample) from ChaCha8 seeded with `seed`; `power_iters`
//!    subspace iterations, re-orthonormalising after every product and keeping every
//!    cells x b block, so the basis K spans the Krylov space (width b x (power_iters + 1));
//!    then Rayleigh-Ritz on range(K) through the width x width Gram matrix `K^T K` and the
//!    `Z^T y` products the iteration already computed (see `block_krylov_svd`).
//! 3. Embeddings `K C diag(d)`, loadings `U_b`, all in f64, returned as f32.
//!
//! # Memory
//!
//! Besides the borrowed X: the cells x width f64 basis K (2.2 KB per cell at the defaults,
//! b = 40 and width 280, against 16 KB per cell for `ScaleData`'s dense 2,000-HVG matrix),
//! one cells x b block, a few genes x width matrices, per-task chunk buffers, and the
//! `Z^T U` accumulators (see `operator.rs`). No transposed copy of X.
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

use operator::{Blocks, Clipped, ScaledOp};

/// The implicit operator `Z` as the SVD sees it: `Z v` and `Z^T u` on
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
    /// Extra random directions per Krylov block (block width `n_comps + oversample`).
    /// Default 10 (see `Default`).
    pub oversample: usize,
    /// Power iterations, i.e. Krylov blocks beyond the first. Default 6.
    pub power_iters: usize,
    /// RNG seed (Seurat `seed.use`; pipeline default 0). Same seed, same result.
    pub seed: u64,
    /// Divide genes by their standard deviation (Seurat `do.scale = TRUE`).
    pub scale: bool,
    /// Seurat's `scale.max` clip (default 10), applied exactly as a sparse correction to the
    /// implicit operator; `None` or `inf` leaves the scaled values unclipped.
    pub clip: Option<f64>,
}

/// Defaults chosen on PBMC 3k against Seurat 5.6 (Seurat's HVGs, clipped `RunPCA`, kNN
/// overlap on the top 10 PCs, k = 20): block Krylov with `oversample = 10, power_iters = 6`
/// gives 0.977 (exact SVD: 0.9975) in 1.1 s at 100k cells on 12 threads, where the plain
/// randomised SVD needed `oversample = 60, power_iters = 30` (6.4 s) for the same accuracy
/// and gave 0.905 at `30 / 10` (1.4 s). PCs 9-11 there have nearly equal singular values,
/// which is the case Krylov methods converge on fastest.
impl Default for PcaParams {
    fn default() -> Self {
        Self { n_comps: 30, oversample: 10, power_iters: 6, seed: 0, scale: true, clip: Some(10.0) }
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
/// - The Krylov width `(n_comps + oversample) x (power_iters + 1)` is capped at `min(n_cells, n_genes)`.
/// - `params.clip` must be positive (`inf` allowed); finite values clip scaled values above it.
/// - Errors: `InvalidParameter` for `n_comps == 0`, `n_comps >= min(n_cells, n_genes)` or a
///   non-positive / NaN `clip`; `InvalidInput` for malformed CSR structure (including
///   duplicate column indices within a row) or non-finite values.
pub fn implicit_scaled_pca<I: Idx, T: Real>(x: CsrView<'_, I, T>, params: &PcaParams) -> Result<PcaResult> {
    implicit_scaled_pca_on(x, params, Device::Cpu)
}

/// [`implicit_scaled_pca`] on a chosen [`Device`] (docs/scfast-integration.md Phase 6).
///
/// `Device::Cpu` is exactly [`implicit_scaled_pca`]. `Device::Cuda(i)` runs the two sparse
/// products of every Krylov step on the GPU (NVRTC CSR kernels with the mean / 1/sd
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

    let b = k.saturating_add(params.oversample).min(n.min(p));
    let clip = params.clip.filter(|c| c.is_finite());
    let (q, ub, d, vb) = match device {
        Device::Cpu => {
            let op = ScaledOp::new(x, &stats.mean, &inv_sd, blocks);
            match clip {
                Some(c) => block_krylov_svd(&Clipped::new(op, &x, &stats.mean, &inv_sd, c), b, params.power_iters, params.seed)?,
                None => block_krylov_svd(&op, b, params.power_iters, params.seed)?,
            }
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(ordinal) => {
            let op = cuda::CudaScaledOp::new(x, &stats.mean, &inv_sd, b, ordinal)?;
            match clip {
                Some(c) => block_krylov_svd(&Clipped::new(op, &x, &stats.mean, &inv_sd, c), b, params.power_iters, params.seed)?,
                None => block_krylov_svd(&op, b, params.power_iters, params.seed)?,
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

/// Block Krylov SVD (Musco & Musco 2015): subspace iteration that keeps every
/// power-iteration block, so the basis spans `[Z Ω, (Z Z^T) Z Ω, ...]` and the
/// top singular vectors converge much faster when singular values are close.
///
/// Returns the raw basis K (cells x width, each block orthonormal), the left singular
/// vectors `U_b`, the singular values and the coefficients C (width x r) of the right
/// singular vectors in K, so the embeddings are `K C diag(d)`.
///
/// K is never orthonormalised as a whole. Every block's `Z^T y` is needed for the next
/// block anyway, so `Z^T K` costs one extra product; with `K^T K = E Λ E^T` the columns of
/// `K T`, `T = E Λ^{-1/2}`, are an orthonormal basis of range(K), and `Z^T (K T) = (Z^T K) T`.
/// Directions with tiny `λ` (blocks that became nearly dependent) are dropped. At least
/// `b` remain: K contains an orthonormal block of width `b`, so `K^T K` has `b`
/// eigenvalues of at least 1.
fn block_krylov_svd<O: Operator>(
    op: &O,
    b: usize,
    iters: usize,
    seed: u64,
) -> Result<(faer::Mat<f64>, faer::Mat<f64>, Vec<f64>, faer::Mat<f64>)> {
    let (n, p) = (op.nrows(), op.ncols());
    let blocks = (iters + 1).min(n.min(p) / b);
    let width = blocks * b;
    let mut k = faer::Mat::<f64>::zeros(n, width);
    let mut ztk = faer::Mat::<f64>::zeros(p, width);
    let omega = dense::gaussian(p, b, seed);
    let mut y = faer::Mat::<f64>::zeros(n, b);
    op.apply(omega.as_ref(), y.as_mut())?;
    drop(omega);
    dense::orthonormalize(y.as_mut());
    // A basis capped by min(n, p) cannot hold every block: the surplus iterations refine
    // the start block instead.
    for _ in blocks..iters + 1 {
        let mut g = op.apply_t(y.as_ref())?;
        dense::orthonormalize(g.as_mut());
        op.apply(g.as_ref(), y.as_mut())?;
        dense::orthonormalize(y.as_mut());
    }
    for i in 0..blocks {
        k.as_mut().submatrix_mut(0, i * b, n, b).copy_from(y.as_ref());
        let mut g = op.apply_t(y.as_ref())?;
        ztk.as_mut().submatrix_mut(0, i * b, p, b).copy_from(g.as_ref());
        if i + 1 < blocks {
            dense::orthonormalize(g.as_mut());
            op.apply(g.as_ref(), y.as_mut())?;
            dense::orthonormalize(y.as_mut());
        }
    }
    let t = dense::whitening(dense::gram(k.as_ref()).as_ref())?;
    let (ub, d, vb) = dense::thin_svd(dense::mul(ztk.as_ref(), t.as_ref()).as_ref())?;
    Ok((k, ub, d, dense::mul(t.as_ref(), vb.as_ref())))
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
