//! Implicitly scaled PCA: replaces Seurat's `ScaleData` + `RunPCA` without ever building
//! the dense scaled matrix (docs/RUST_ENGINE.md sections 1 and 8).
//!
//! With X the log-normalised cells x genes matrix (HVGs only), per-gene mean `mu` and
//! standard deviation `sigma`, the scaled matrix is `Z = (X - 1 mu^T) diag(1/sigma)`.
//! Randomised SVD only needs `Z v` and `Z^T u`, which are sparse products plus a rank-one
//! correction, so memory stays at the size of the sparse X.
//!
//! STATUS: stub. Signature and contract are fixed; the body is implemented separately.

use crate::error::Result;
use crate::types::{CsrView, Idx, Real, RowMajor};

/// Parameters, with Seurat-compatible defaults.
#[derive(Clone, Debug)]
pub struct PcaParams {
    /// Number of components (Seurat `npcs`; pipeline default 30).
    pub n_comps: usize,
    /// Extra random directions for the range finder (Halko et al.); 10 is standard.
    pub oversample: usize,
    /// Power iterations; more improves accuracy when the spectrum decays slowly.
    pub power_iters: usize,
    /// RNG seed (Seurat `seed.use`; pipeline default 0). Same seed, same result.
    pub seed: u64,
    /// Divide genes by their standard deviation (Seurat `do.scale = TRUE`).
    pub scale: bool,
    /// Seurat's `scale.max` clip (default 10). The implicit operator cannot clip, so this is
    /// recorded for reference only and the differences are measured by the parity checks.
    pub clip: Option<f64>,
}

impl Default for PcaParams {
    fn default() -> Self {
        Self { n_comps: 30, oversample: 10, power_iters: 4, seed: 0, scale: true, clip: Some(10.0) }
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
pub fn implicit_scaled_pca<I: Idx, T: Real>(x: CsrView<'_, I, T>, params: &PcaParams) -> Result<PcaResult> {
    let _ = (x, params);
    unimplemented!("implicit_scaled_pca: Phase 3")
}
