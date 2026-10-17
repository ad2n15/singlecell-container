//! Leiden clustering: replaces Seurat's `FindClusters(algorithm = 4)`.
//!
//! CPU: single-clustering 0.7.0 (pinned), RB objective, which is what Seurat
//! (`RBConfigurationVertexPartition`) and Scanpy optimise (docs/RUST_ENGINE.md section 4).
//! GPU (Phase 7): cuGraph Leiden behind the same `Clusterer` trait.
//!
//! # Self-loops (the SNN diagonal)
//!
//! Seurat's `RNA_snn` keeps a diagonal of 1s, and how a self-loop enters the RB null model
//! depends on how the graph is handed to leidenalg:
//!
//! - Seurat 5.0.1 (verified: `RunLeiden` -> `leiden` R package -> python-igraph
//!   `Graph.Weighted_Adjacency`) converts the SNN matrix in igraph's default **directed**
//!   mode: every stored off-diagonal weight becomes an arc, so each edge is two arcs, and
//!   every diagonal 1 becomes **one** loop arc. (Seurat 5.1+ hands the matrix to `leidenbase`
//!   through `igraph::graph_from_adjacency_matrix`, also directed by default; not verified
//!   here, as only 5.0.1 is installed.) leidenalg's directed RB is
//!   `sum_c [w_c - g * K_c^out * K_c^in / m]`, so each loop adds 1 to a node's strength and
//!   1 to m.
//! - single-clustering is undirected with the igraph convention: a loop of weight w adds 2w
//!   to the node's strength and w to the total weight.
//!
//! Passing the diagonal at **half weight** makes the two objectives identical: with an
//! undirected loop of 0.5 the strength gains 1 and 4m = 2 * (2M + n), so
//! `Q_undirected = Q_directed / 2` up to a constant, for every partition. Keeping the
//! diagonal at full weight (strength + 2) or dropping it (strength + 0) optimises a
//! different objective from Seurat's. [`DIAGONAL_SCALE`] records the choice; the evidence
//! is in its documentation.

use std::cmp::Reverse;

use single_clustering::community_search::leiden::{leiden as sc_leiden, LeidenConfig, ObjectiveKind};
use single_clustering::network::CSRNetwork;
use single_clustering::ClusteringError;

use crate::error::{Error, Result};
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

/// Factor applied to diagonal (self-loop) weights before they reach single-clustering.
///
/// 0.5 reproduces the objective Seurat's `FindClusters(algorithm = 4)` optimises on its SNN
/// (derivation in the module documentation). Graphs without a diagonal (scanpy
/// connectivities, or `SnnParams { keep_diagonal: false }`) are unaffected.
///
/// Evidence (17 Oct 2026, Seurat 5.0.1 + leidenalg 0.12 / python-igraph 1.0, 2-core VM):
///
/// - Same objective, numerically: for leidenalg's own partitions, RB quality on Seurat's
///   directed graph equals RB quality on the undirected half-loop graph to ~1e-12 relative
///   (PBMC 3k: 29982.8479199 vs 29982.8479199).
/// - Real PBMC 3k (2,638 cells, dims 1:10, k 20, Seurat's own SNN), 10 seeds per method,
///   mean cluster count:
///
///   | resolution | leidenalg as Seurat runs it | diag x 0.5 | diag x 1 | diag dropped |
///   |------------|-----------------------------|------------|----------|--------------|
///   | 0.5        | 8.7                         | 9.2        | 9.1      | 9.0          |
///   | 1.0        | 11.4                        | 11.3       | 12.0     | 11.1         |
///   | 2.0        | 17.7                        | 17.3       | 18.1     | 15.5         |
///
///   Dropping the diagonal under-splits and full weight over-splits as resolution grows;
///   half weight tracks Seurat at every resolution. Mean ARI against the leidenalg runs at
///   2.0: 0.789 (x 0.5), 0.791 (x 1), 0.744 (dropped), vs 0.762 between leidenalg's own seeds.
///   At the pipeline resolution 0.5 all three are within noise (ARI 0.889-0.896 vs leidenalg,
///   whose own seed-to-seed ARI is 0.851).
/// - Synthetic 2k / 5k bench data and the 500-cell fixture: every variant and seed gives
///   ARI 1.0 against Seurat's labels.
pub const DIAGONAL_SCALE: f32 = 0.5;

/// single-clustering Leiden with the RB objective.
pub struct SingleClusteringLeiden;

impl Clusterer for SingleClusteringLeiden {
    fn name(&self) -> &'static str {
        "single-clustering 0.7.0 (Leiden, RB)"
    }

    /// Errors: malformed CSR or a graph that isn't symmetric (`InvalidInput`), negative or
    /// non-finite resolution or `n_iterations == 0` (`InvalidParameter`).
    fn cluster(&self, graph: &SparseGraph, params: &ClusterParams) -> Result<Vec<u32>> {
        if !params.resolution.is_finite() || params.resolution < 0.0 {
            return Err(Error::InvalidParameter(format!(
                "resolution must be finite and non-negative, got {}",
                params.resolution
            )));
        }
        if params.n_iterations == 0 {
            return Err(Error::InvalidParameter("n_iterations must be at least 1".into()));
        }
        validate_shape(graph)?;
        if graph.n == 0 {
            return Ok(Vec::new());
        }

        // from_csr_parts takes owned buffers: this is the one copy of the graph per call
        let mut weights = graph.weights.clone();
        if DIAGONAL_SCALE != 1.0 {
            for i in 0..graph.n {
                let (s, e) = (graph.indptr[i], graph.indptr[i + 1]);
                for (w, &j) in weights[s..e].iter_mut().zip(&graph.indices[s..e]) {
                    if j as usize == i {
                        *w *= DIAGONAL_SCALE;
                    }
                }
            }
        }
        let network = CSRNetwork::from_csr_parts(graph.indptr.clone(), graph.indices.clone(), weights, None)
            .map_err(map_err)?;

        let config = LeidenConfig {
            objective: ObjectiveKind::Rb { resolution: params.resolution },
            seed: Some(params.seed),
            n_iterations: params.n_iterations,
            refine: true,
            ..Default::default()
        };
        let clustering = sc_leiden(&network, &config).map_err(map_err)?;
        Ok(relabel_by_size(clustering.labels()))
    }
}

/// Convenience: CPU Leiden with the default backend.
pub fn leiden(graph: &SparseGraph, params: &ClusterParams) -> Result<Vec<u32>> {
    SingleClusteringLeiden.cluster(graph, params)
}

/// Structural checks single-clustering would otherwise trip over (it derives n from
/// `indptr`, so a wrong `n` would go unnoticed). Index bounds, weight sign/finiteness and
/// symmetry are checked by `CSRNetwork::from_csr_parts`.
fn validate_shape(g: &SparseGraph) -> Result<()> {
    if g.indptr.len() != g.n + 1 {
        return Err(Error::InvalidInput(format!("indptr has {} entries, expected n + 1 = {}", g.indptr.len(), g.n + 1)));
    }
    if g.indices.len() != g.weights.len() {
        return Err(Error::InvalidInput(format!(
            "indices ({}) and weights ({}) differ in length",
            g.indices.len(),
            g.weights.len()
        )));
    }
    if g.indptr[0] != 0 || g.indptr[g.n] != g.indices.len() {
        return Err(Error::InvalidInput("indptr must start at 0 and end at nnz".into()));
    }
    if g.indptr.windows(2).any(|w| w[1] < w[0]) {
        return Err(Error::InvalidInput("indptr is not non-decreasing".into()));
    }
    Ok(())
}

fn map_err(e: ClusteringError) -> Error {
    match e {
        ClusteringError::InvalidConfig(m) => Error::InvalidParameter(m),
        ClusteringError::NodeIndexOutOfRange { .. }
        | ClusteringError::NonFiniteWeight { .. }
        | ClusteringError::NegativeWeight { .. }
        | ClusteringError::InvalidCsr(_)
        | ClusteringError::AsymmetricGraph { .. }
        | ClusteringError::TooManyNodes { .. } => Error::InvalidInput(format!("graph rejected by single-clustering: {e}")),
        other => Error::Backend { context: "single-clustering Leiden", message: other.to_string() },
    }
}

/// Renumber clusters 0.. by decreasing size, ties broken by the lowest node index in the
/// cluster, so labels don't depend on the backend's internal numbering.
fn relabel_by_size(labels: &[usize]) -> Vec<u32> {
    let n_slots = labels.iter().copied().max().map_or(0, |m| m + 1);
    let mut size = vec![0usize; n_slots];
    let mut first = vec![usize::MAX; n_slots];
    for (v, &c) in labels.iter().enumerate() {
        size[c] += 1;
        first[c] = first[c].min(v);
    }
    let mut order: Vec<usize> = (0..n_slots).filter(|&c| size[c] > 0).collect();
    order.sort_unstable_by_key(|&c| (Reverse(size[c]), first[c]));
    let mut new_id = vec![0u32; n_slots];
    for (rank, &c) in order.iter().enumerate() {
        new_id[c] = rank as u32;
    }
    labels.iter().map(|&c| new_id[c]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relabels_by_size_then_first_index() {
        // sizes: a=2 (first 0), b=3 (first 1), c=2 (first 3), d=1
        let raw = [7, 2, 2, 5, 7, 2, 5, 9];
        assert_eq!(relabel_by_size(&raw), vec![1, 0, 0, 2, 1, 0, 2, 3]);
        assert!(relabel_by_size(&[]).is_empty());
    }

    #[test]
    fn rejects_bad_shapes_and_params() {
        let g = SparseGraph { n: 2, indptr: vec![0, 1, 2], indices: vec![1, 0], weights: vec![1.0, 1.0] };
        assert!(leiden(&g, &ClusterParams::default()).is_ok());
        let bad_n = SparseGraph { n: 3, ..g.clone() };
        assert!(matches!(leiden(&bad_n, &ClusterParams::default()), Err(Error::InvalidInput(_))));
        let asym = SparseGraph { n: 2, indptr: vec![0, 1, 1], indices: vec![1], weights: vec![1.0] };
        assert!(matches!(leiden(&asym, &ClusterParams::default()), Err(Error::InvalidInput(_))));
        let p = ClusterParams { n_iterations: 0, ..Default::default() };
        assert!(matches!(leiden(&g, &p), Err(Error::InvalidParameter(_))));
        let p = ClusterParams { resolution: f64::NAN, ..Default::default() };
        assert!(matches!(leiden(&g, &p), Err(Error::InvalidParameter(_))));
    }
}
