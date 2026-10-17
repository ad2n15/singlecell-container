//! Shared-nearest-neighbour graph: replaces the SNN part of Seurat's `FindNeighbors`
//! (`ComputeSNN`), which is the graph `FindClusters` clusters on.
//!
//! Seurat's definition: with each cell's neighbour set N(i) of size k INCLUDING itself,
//! the edge weight is the Jaccard index |N(i) & N(j)| / |N(i) | N(j)| =
//! s / (2k - s) with s = |N(i) & N(j)|, and weights below `prune` (default 1/15) are set to
//! zero. The result is symmetric and includes the diagonal (weight 1).
//!
//! STATUS: stub. Signature and contract are fixed; the body is implemented separately.

use crate::error::Result;
use crate::types::{Knn, SparseGraph};

#[derive(Clone, Debug)]
pub struct SnnParams {
    /// Seurat `prune.SNN` (default 1/15).
    pub prune: f64,
    /// Keep the diagonal (self weight 1), as Seurat's `RNA_snn` does.
    pub keep_diagonal: bool,
}

impl Default for SnnParams {
    fn default() -> Self {
        Self { prune: 1.0 / 15.0, keep_diagonal: true }
    }
}

/// Build the SNN graph from a kNN result (self included, as returned by `knn::knn`).
/// Output: full symmetric CSR (both directions stored), rows with sorted column indices.
pub fn snn_from_knn(knn: &Knn, params: &SnnParams) -> Result<SparseGraph> {
    let _ = (knn, params);
    unimplemented!("snn_from_knn: Phase 4")
}
