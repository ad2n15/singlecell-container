//! Leiden clustering: replaces Seurat's `FindClusters(algorithm = 4)`.
//!
//! CPU: single-clustering 0.7.0 (pinned), RB objective, which is what Seurat
//! (`RBConfigurationVertexPartition`) and Scanpy optimise (docs/RUST_ENGINE.md section 4).
//! GPU (Phase 7): cuGraph Leiden behind the same `Clusterer` trait.
//!
//! STATUS: stub. Signature and contract are fixed; the body is implemented separately.

use crate::error::Result;
use crate::types::SparseGraph;

/// Parameter mapping from Seurat / the pipeline (docs/RUST_ENGINE.md section 4).
#[derive(Clone, Debug)]
pub struct ClusterParams {
    /// Seurat `resolution` (pipeline default 0.5).
    pub resolution: f64,
    /// Seurat `random.seed` (pipeline default 0).
    pub seed: u64,
    /// Seurat `n.iter` / leidenalg `n_iterations` (pipeline default 2).
    pub n_iterations: usize,
}

impl Default for ClusterParams {
    fn default() -> Self {
        Self { resolution: 0.5, seed: 0, n_iterations: 2 }
    }
}

/// A clustering backend. CPU (single-clustering) now; cuGraph in Phase 7.
pub trait Clusterer {
    fn name(&self) -> &'static str;
    /// One label per node, 0..n_clusters-1, numbered by decreasing cluster size (ties by
    /// lowest node index), like Seurat's `seurat_clusters` levels.
    fn cluster(&self, graph: &SparseGraph, params: &ClusterParams) -> Result<Vec<u32>>;
}

/// single-clustering Leiden with the RB objective.
pub struct SingleClusteringLeiden;

impl Clusterer for SingleClusteringLeiden {
    fn name(&self) -> &'static str {
        "single-clustering 0.7.0 (Leiden, RB)"
    }
    fn cluster(&self, graph: &SparseGraph, params: &ClusterParams) -> Result<Vec<u32>> {
        let _ = (graph, params);
        unimplemented!("SingleClusteringLeiden: Phase 2")
    }
}

/// Convenience: CPU Leiden with the default backend.
pub fn leiden(graph: &SparseGraph, params: &ClusterParams) -> Result<Vec<u32>> {
    SingleClusteringLeiden.cluster(graph, params)
}
