"""Type stubs for scfast._scfast, the native (Rust, PyO3) core of scfast.

These are the low-level entry points behind scfast.pca / knn / snn / leiden. They take numpy
arrays that the Python layer has already checked and laid out (C-contiguous, native byte
order, supported dtypes) and do no scipy handling. Every function releases the GIL while
it works. Error mapping:

- invalid input or parameter (scfast_core::Error::InvalidInput / InvalidParameter) -> ValueError
- failure inside a dependency (scfast_core::Error::Backend) -> RuntimeError
- a Rust panic -> RuntimeError; an unimplemented kernel -> NotImplementedError
"""

from typing import Any

import numpy as np
import numpy.typing as npt

__version__: str
"""Version of the Rust crate (workspace version in rust/Cargo.toml)."""

def pca_csr(
    n_rows: int,
    n_cols: int,
    indptr: npt.NDArray[np.int32] | npt.NDArray[np.int64],
    indices: npt.NDArray[np.int32] | npt.NDArray[np.int64],
    data: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    n_comps: int,
    seed: int,
    scale: bool,
    oversample: int,
    power_iters: int,
    /,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Implicitly scaled PCA of a CSR matrix (cells x genes) given as raw arrays.

    The three arrays are borrowed without copying: indptr and indices must share one dtype
    (int32 or int64), data is float32 or float64, all C-contiguous. Rows must be canonical
    (strictly increasing column indices).

    Returns (embeddings (n_rows, k), loadings (n_cols, k), stdev (k,)), all float32.
    """

def csr_view_info(
    n_rows: int,
    n_cols: int,
    indptr: npt.NDArray[Any],
    indices: npt.NDArray[Any],
    data: npt.NDArray[Any],
    /,
) -> dict[str, Any]:
    """Run pca_csr's dtype dispatch and validation without the kernel.

    Returns {"index_type": "i32" | "i64", "value_type": "f32" | "f64", "indptr_ptr",
    "indices_ptr", "data_ptr" (addresses of the borrowed buffers), "n_rows", "n_cols", "nnz"}.
    Used by the tests to prove the CSR input is read in place.
    """

def knn(
    x: npt.NDArray[np.float32],
    k: int,
    method: str,
    seed: int,
    auto_threshold: int,
    m: int,
    ef_construction: int,
    ef_search: int,
    /,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """k nearest neighbours of each row of a C-contiguous float32 (n, d) array.

    method is "exact", "hnsw" or "auto" (exact below auto_threshold rows, else HNSW with
    m / ef_construction / ef_search). Non-finite values raise ValueError.

    Returns (indices int32 (n, k), distances float32 (n, k)), the row itself first.
    """

def snn(
    indices: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint32] | npt.NDArray[np.uint64],
    prune: float,
    keep_diagonal: bool,
    /,
) -> tuple[npt.NDArray[np.int32] | npt.NDArray[np.int64], npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Shared-nearest-neighbour (Jaccard) graph from C-contiguous (n, k) kNN indices.

    Returns the CSR arrays (indptr, indices, weights) of the full symmetric n x n graph;
    indptr is int32 when the number of stored entries fits, else int64.
    """

def leiden_csr(
    n: int,
    indptr: npt.NDArray[np.int32] | npt.NDArray[np.int64],
    indices: npt.NDArray[np.int32] | npt.NDArray[np.int64],
    data: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    resolution: float,
    seed: int,
    n_iterations: int,
    /,
) -> npt.NDArray[np.int32]:
    """Leiden (RB objective) on a square, symmetric, canonical CSR graph given as raw arrays.

    The graph is copied once into the clustering library's own buffers. Returns int32
    labels numbered by decreasing cluster size.
    """

def _probe_detach(seconds: float, /) -> float:
    """Busy-wait for `seconds` with the GIL released, through the kernels' own wrapper.

    Returns the measured duration. Test helper for GIL release.
    """

def _probe_error(kind: str, message: str, /) -> None:
    """Raise what a scfast_core::Error of `kind` ("invalid_input", "invalid_parameter",
    "backend") maps to. Test helper for error mapping."""

def _probe_panic(kind: str, /) -> None:
    """Panic inside the kernel wrapper: "panic", "unimplemented" (like a stub kernel) or
    "rayon" (on a rayon worker thread). Test helper: must raise, never abort."""
