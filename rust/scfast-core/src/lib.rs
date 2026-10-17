//! scfast-core: the numerical core of the scfast engine (docs/RUST_ENGINE.md).
//!
//! Kernels, each callable on its own so Seurat (via scfast-r) and Python (via scfast-py)
//! can swap in one step at a time:
//!
//! | Module      | Replaces                         | Function                     |
//! |-------------|----------------------------------|------------------------------|
//! | [`pca`]     | `ScaleData` + `RunPCA`           | [`pca::implicit_scaled_pca`] |
//! | [`knn`]     | kNN part of `FindNeighbors`      | [`knn::knn`]                 |
//! | [`snn`]     | SNN part of `FindNeighbors`      | [`snn::snn_from_knn`]        |
//! | [`cluster`] | `FindClusters(algorithm = 4)`    | [`cluster::leiden`]          |
//!
//! Everything here is the CPU reference implementation; the CUDA backend (Phase 6) is
//! tested against it.

pub mod cluster;
pub mod error;
pub mod knn;
pub mod pca;
pub mod snn;
pub mod types;

pub use error::{Error, Result};
pub use types::{CsrView, Idx, Knn, Real, RowMajor, SparseGraph};
