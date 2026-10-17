//! k nearest neighbours on PC embeddings: replaces the kNN part of Seurat's
//! `FindNeighbors` (Annoy, approximate) with exact search for moderate sizes and HNSW for
//! large ones.
//!
//! STATUS: stub. Signature and contract are fixed; the body is implemented separately.

use crate::error::Result;
use crate::types::{Knn, RowMajor};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KnnMethod {
    /// Exact brute force (blocked, parallel). O(n^2 d): fine up to roughly 50k-100k cells.
    Exact,
    /// Approximate HNSW (hnsw_rs). `m` = links per node, `ef_construction`, `ef_search`.
    Hnsw { m: usize, ef_construction: usize, ef_search: usize },
    /// Exact below `auto_threshold` cells, HNSW above.
    Auto,
}

#[derive(Clone, Debug)]
pub struct KnnParams {
    /// Neighbours per cell INCLUDING the cell itself (Seurat `k.param`, default 20).
    pub k: usize,
    pub method: KnnMethod,
    /// Cell count where `Auto` switches from exact to HNSW. Set from the bench scaling curve.
    pub auto_threshold: usize,
    /// Seed for HNSW construction (exact search ignores it).
    pub seed: u64,
}

impl Default for KnnParams {
    fn default() -> Self {
        Self { k: 20, method: KnnMethod::Auto, auto_threshold: 50_000, seed: 0 }
    }
}

/// k nearest neighbours (Euclidean) of every row of `x` (cells x dims, e.g. the first 30
/// PCs). Returns `Knn` with the cell itself first (distance 0) and the remaining k-1
/// neighbours in increasing distance; ties broken by lower index so results are
/// deterministic.
pub fn knn(x: &RowMajor, params: &KnnParams) -> Result<Knn> {
    let _ = (x, params);
    unimplemented!("knn: Phase 4")
}
