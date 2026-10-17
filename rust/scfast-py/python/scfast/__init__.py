"""scfast: fast single-cell kernels in Rust for Seurat- and Scanpy-style workflows.

The steps from raw counts to cluster labels (docs/RUST_ENGINE.md section 5)::

    import scfast
    pre = scfast.preprocess(X_counts_csr, n_hvg=2000)                   # normalise + vst HVGs, one pass
    pcs, loadings, sdev = scfast.pca(X_csr, n_comps=30, seed=0)       # no dense scaled matrix
    knn_idx, knn_dist   = scfast.knn(pcs, k=20)                         # self first, Euclidean
    snn                 = scfast.snn(knn_idx, prune=1/15)               # scipy CSR, Jaccard
    labels              = scfast.leiden(snn, resolution=0.5, seed=0, n_iterations=2)

Every call releases the GIL while the Rust kernel runs. Errors from the kernels arrive as
``ValueError`` (bad input or parameter) or ``RuntimeError`` (internal failure); a kernel
that is not implemented in this build raises ``NotImplementedError``.

Threads: the kernels use rayon's global pool, sized by ``RAYON_NUM_THREADS`` (default: all
cores). Inputs must not be modified from another thread while a call is running.
"""

from __future__ import annotations

import math
import operator
from typing import Any, Literal

import numpy as np

from . import _scfast
from ._scfast import __version__

__all__ = ["__version__", "device_info", "knn", "leiden", "leiden_device", "pca", "preprocess", "snn", "umap"]

_I32_MAX = int(np.iinfo(np.int32).max)
_U64_MAX = int(np.iinfo(np.uint64).max)
_INDEX_DTYPES = (np.dtype(np.int32), np.dtype(np.int64))
_VALUE_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
_KNN_INDEX_DTYPES = (np.dtype(np.int32), np.dtype(np.int64), np.dtype(np.uint32), np.dtype(np.uint64))


# ---------------------------------------------------------------- argument checks

def _int(name: str, value: Any, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        v = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {v}")
    if maximum is not None and v > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {v}")
    return v


def _float(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a number, got bool")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    return v


def _seed(seed: Any) -> int:
    return _int("seed", seed, 0, _U64_MAX)


def _resolve_device(device: Any) -> tuple[str, str]:
    """(requested, resolved) for a PCA / kNN device string (Phase 6).

    ``"auto"`` resolves to ``"cuda:0"`` only when this build has the ``cuda`` feature and a
    device is usable, else ``"cpu"``. An unknown string is ``ValueError``; ``"cuda"`` /
    ``"cuda:<i>"`` without a usable device is ``RuntimeError`` (message starting ``GPU:``).
    """
    if not isinstance(device, str):
        raise ValueError(f"device must be 'auto', 'cpu', 'cuda' or 'cuda:<index>', got {device!r}")
    requested = device.strip().lower()
    try:
        resolved = _scfast.device(requested)
    except ValueError:
        raise ValueError(f"device must be 'auto', 'cpu', 'cuda' or 'cuda:<index>', got {device!r}") from None
    except RuntimeError as e:
        raise RuntimeError(f"device={device!r}: {e}") from None
    return requested, resolved


def _with_cpu_fallback(requested: str, resolved: str, call: Any) -> Any:
    """Run ``call(resolved)``; when the device came from ``"auto"`` and the GPU fails
    (``RuntimeError`` whose message carries ``GPU:``, e.g. out of device memory), warn and
    run again on the CPU. An explicit device is never retried."""
    if resolved == "cpu" or requested != "auto":
        return call(resolved)
    try:
        return call(resolved)
    except RuntimeError as e:
        if "GPU: " not in str(e):
            raise
        import warnings

        warnings.warn(f"scfast: CUDA failed, falling back to the CPU: {e}", RuntimeWarning, stacklevel=3)
        return call("cpu")


def device_info() -> dict[str, Any]:
    """CUDA state of this build (never raises).

    Keys: ``cuda_built`` (compiled with the ``cuda`` feature), ``cuda_available`` (driver,
    NVRTC and cuBLAS load and a device is visible), ``device_count``, ``device_name``,
    ``free_gb``, ``total_gb``, ``compute_capability`` (device 0; ``None`` when unavailable),
    ``cuda_error`` (why CUDA is unavailable, else ``None``) and ``auto_device`` (what
    ``device="auto"`` resolves to: ``"cpu"`` or ``"cuda:0"``).
    """
    return dict(_scfast.device_info())


def _describe(x: Any) -> str:
    fmt = getattr(x, "format", None)
    name = type(x).__name__
    return f"{name} (format {fmt!r})" if isinstance(fmt, str) else name


def _csr_parts(X: Any, what: str) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Shape and (indptr, indices, data) of a scipy CSR matrix, ready for the Rust side.

    float32/float64 data and int32/int64 indices are passed through untouched, so Rust
    borrows scipy's own buffers. Anything else is converted (a copy) as documented in pca().
    Structure and canonical order are checked in Rust (O(nnz), GIL released).
    """
    import scipy.sparse as sp

    if not sp.issparse(X) or X.format != "csr":
        hint = f" Convert it with {what}.tocsr() first (a copy)." if sp.issparse(X) else ""
        raise TypeError(
            f"{what} must be a scipy.sparse CSR matrix (csr_matrix or csr_array), got {_describe(X)}.{hint}"
        )
    if len(X.shape) != 2:
        raise ValueError(f"{what} must be 2-D, got shape {X.shape}")
    n_rows, n_cols = (int(s) for s in X.shape)

    indptr, indices, data = X.indptr, X.indices, X.data
    if indptr.dtype not in _INDEX_DTYPES or indices.dtype not in _INDEX_DTYPES or indptr.dtype != indices.dtype:
        # scipy always gives both the same int32/int64 dtype; this only handles hand-made ones.
        if indices.dtype == np.int32 and indptr.size and int(indptr[-1]) <= _I32_MAX:
            indptr = indptr.astype(np.int32)  # n_rows + 1 values: cheap
        else:
            indptr, indices = indptr.astype(np.int64), indices.astype(np.int64)
    if data.dtype not in _VALUE_DTYPES:
        if data.dtype.kind not in "biuf":
            raise TypeError(f"{what} must hold real numbers, got dtype {data.dtype}")
        data = data.astype(np.float64 if data.dtype.kind == "f" and data.dtype.itemsize > 8 else np.float32)
    # no-ops (same buffer) for scipy's own contiguous, native-endian arrays
    indptr = np.ascontiguousarray(indptr, dtype=indptr.dtype.newbyteorder("="))
    indices = np.ascontiguousarray(indices, dtype=indices.dtype.newbyteorder("="))
    data = np.ascontiguousarray(data, dtype=data.dtype.newbyteorder("="))
    return n_rows, n_cols, indptr, indices, data


def _inspect_csr(X: Any) -> dict[str, Any]:
    """Diagnostics: the Rust types and buffer addresses pca() would use for X (no kernel)."""
    return _scfast.csr_view_info(*_csr_parts(X, "X"))


# ---------------------------------------------------------------- public API

def pca(
    X: Any,
    n_comps: int = 30,
    seed: int = 0,
    scale: bool = True,
    oversample: int = 10,
    power_iters: int = 6,
    *,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    clip: float | None = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implicitly scaled PCA, replacing Seurat's ``ScaleData`` + ``RunPCA``.

    Centres (and with ``scale=True`` scales to unit variance) each gene implicitly and runs
    a block Krylov SVD, so the dense cells x genes scaled matrix is never built.

    Parameters
    ----------
    X
        scipy.sparse CSR matrix (``csr_matrix`` or ``csr_array``), cells x genes, usually
        the log-normalised HVG matrix. float32/float64 data with int32/int64 indices is
        read in place (zero copy); other real dtypes are converted to float32 first (a copy).
        Must be canonical (sorted, unique column indices per row); call ``X.sum_duplicates()``
        if not. Other sparse formats and dense arrays raise ``TypeError``.
    n_comps
        Number of components (Seurat ``npcs``).
    seed
        RNG seed for the SVD start block; the same seed gives the same result.
    scale
        Divide each gene by its standard deviation (n - 1 denominator), as ``ScaleData``.
    oversample, power_iters
        Block Krylov SVD settings: extra directions per block (width ``n_comps +
        oversample``) and iterations (``power_iters + 1`` blocks). The defaults reach
        Seurat parity on PBMC 3k (kNN overlap on the top 10 PCs 0.977; exact SVD 0.9975).
    device
        ``"auto"`` (CUDA when this build has the ``cuda`` feature and a device is usable,
        else CPU), ``"cpu"``, ``"cuda"`` or ``"cuda:<index>"``. With ``"auto"`` a GPU
        failure (e.g. out of device memory) falls back to the CPU with a ``RuntimeWarning``;
        an explicit ``"cuda"`` that cannot be used raises ``RuntimeError``. On CUDA the
        sparse products run on the device in single precision, so results match the CPU to
        the parity thresholds rather than bitwise. See :func:`device_info`.
    clip
        Seurat's ``ScaleData(scale.max)``: scaled values above it are clipped, exactly, as a
        sparse correction to the implicit operator. ``None`` leaves them unclipped.

    Returns
    -------
    embeddings : float32 array (cells, n_comps)
        Cell embeddings U * d (Seurat ``Embeddings``, Scanpy ``obsm["X_pca"]``).
    loadings : float32 array (genes, n_comps)
        Gene loadings V (Seurat ``Loadings``, Scanpy ``varm["PCs"]``).
    stdev : float32 array (n_comps,)
        Component standard deviations d / sqrt(cells - 1) (Seurat ``Stdev``).
    """
    requested, resolved = _resolve_device(device)
    n_rows, n_cols, indptr, indices, data = _csr_parts(X, "X")
    n_comps = _int("n_comps", n_comps, 1)
    if n_rows == 0 or n_cols == 0:
        raise ValueError(f"X is empty (shape {(n_rows, n_cols)})")
    if n_comps > min(n_rows, n_cols):
        raise ValueError(f"n_comps ({n_comps}) must not exceed min(cells, genes) = {min(n_rows, n_cols)}")
    seed, scale = _seed(seed), bool(scale)
    oversample, power_iters = _int("oversample", oversample), _int("power_iters", power_iters)
    return _with_cpu_fallback(requested, resolved, lambda dev: _scfast.pca_csr(
        n_rows, n_cols, indptr, indices, data, n_comps, seed, scale, oversample, power_iters, dev,
        None if clip is None else _float("clip", clip),
    ))


def knn(
    X: Any,
    k: int = 20,
    method: Literal["auto", "exact", "hnsw"] = "auto",
    seed: int = 0,
    auto_threshold: int = 50_000,
    m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 64,
    *,
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """k nearest neighbours (Euclidean), replacing the kNN part of Seurat's ``FindNeighbors``.

    Parameters
    ----------
    X
        Dense 2-D array, cells x dims (e.g. the first 30 PCs). Used as float32 C-order; any
        other real dtype or memory layout is converted first. The kernel takes its own
        copy of the (small) embedding matrix in any case.
    k
        Neighbours per cell INCLUDING the cell itself (Seurat ``k.param``), at least 2.
    method
        ``"exact"`` (blocked brute force, exact), ``"hnsw"`` (approximate, hnsw_rs) or
        ``"auto"``: exact when there are fewer than ``auto_threshold`` cells, else HNSW
        with ``m``, ``ef_construction`` and ``ef_search``.
    seed
        Seeds the HNSW insertion order. Exact results are deterministic; HNSW results are
        not bit-reproducible across runs (hnsw_rs inserts in parallel).
    device
        ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"cuda:<index>"``; see :func:`pca`. On CUDA
        the search is exact brute force (cuBLAS GEMM + top-k) at every size, ``method`` is
        ignored and ``k`` must be at most 129.

    Returns
    -------
    indices : int32 array (cells, k)
        Row i starts with i itself, then the k - 1 nearest other cells by increasing
        distance (ties by lower index).
    distances : float32 array (cells, k)
        Euclidean distances, 0 in the first column.
    """
    import scipy.sparse as sp

    requested, resolved = _resolve_device(device)
    if sp.issparse(X):
        raise TypeError("X must be a dense array (e.g. PCA embeddings), got a scipy sparse matrix")
    a = np.asarray(X)
    if a.ndim != 2:
        raise ValueError(f"X must be 2-D (cells x dims), got {a.ndim}-D with shape {a.shape}")
    if a.dtype.kind not in "biuf":
        raise TypeError(f"X must hold real numbers, got dtype {a.dtype}")
    n, d = a.shape
    if n == 0 or d == 0:
        raise ValueError(f"X is empty (shape {a.shape})")
    k = _int("k", k, 2)  # the cell itself plus at least one neighbour
    if k > n:
        raise ValueError(f"k ({k}) must not exceed the number of cells ({n}); k counts the cell itself")
    if method not in ("auto", "exact", "hnsw"):
        raise ValueError(f"method must be 'auto', 'exact' or 'hnsw', got {method!r}")
    a = np.ascontiguousarray(a, dtype=np.float32)  # no copy when already float32 C-order
    seed, auto_threshold = _seed(seed), _int("auto_threshold", auto_threshold)
    m, ef_construction, ef_search = _int("m", m, 2), _int("ef_construction", ef_construction, 1), _int("ef_search", ef_search, 1)
    return _with_cpu_fallback(requested, resolved, lambda dev: _scfast.knn(
        a, k, method, seed, auto_threshold, m, ef_construction, ef_search, dev,
    ))


def snn(indices: Any, prune: float = 1 / 15, keep_diagonal: bool = True) -> Any:
    """Shared-nearest-neighbour graph, replacing the SNN part of Seurat's ``FindNeighbors``.

    The weight between cells i and j is the Jaccard index of their neighbour sets,
    s / (2k - s) with s = |N(i) & N(j)|, and weights below ``prune`` are dropped
    (Seurat ``prune.SNN``). Any two cells sharing a neighbour are connected, not only kNN
    pairs.

    Parameters
    ----------
    indices
        Integer array (cells, k) of neighbour indices with the cell itself included, as
        returned by :func:`knn` (Seurat's ``Indices()`` also works). Each row is used as the
        neighbour set exactly as given; a cell listed twice in a row is an error.
    prune
        Jaccard cut-off in [0, 1].
    keep_diagonal
        Keep the diagonal (weight 1), as Seurat's ``RNA_snn`` does.

    Returns
    -------
    scipy.sparse.csr_matrix, float32, (cells, cells)
        Symmetric, both directions of every edge stored, sorted column indices.
    """
    import scipy.sparse as sp

    if sp.issparse(indices):
        raise TypeError("indices must be a dense integer array (cells, k), got a scipy sparse matrix")
    idx = np.asarray(indices)
    if idx.ndim != 2:
        raise ValueError(f"indices must be 2-D (cells, k), got {idx.ndim}-D with shape {idx.shape}")
    if idx.dtype.kind not in "iu":
        raise TypeError(f"indices must be an integer array, got dtype {idx.dtype}")
    n, k = idx.shape
    if n == 0 or k == 0:
        raise ValueError(f"indices is empty (shape {idx.shape})")
    if n > _I32_MAX:
        raise ValueError(f"{n} cells exceed the int32 index range")
    prune = _float("prune", prune)
    if not 0.0 <= prune <= 1.0:
        raise ValueError(f"prune must be in [0, 1], got {prune}")
    if idx.dtype not in _KNN_INDEX_DTYPES:
        idx = idx.astype(np.int64)
    idx = np.ascontiguousarray(idx, dtype=idx.dtype.newbyteorder("="))
    indptr, col, weights = _scfast.snn(idx, prune, bool(keep_diagonal))
    return sp.csr_matrix((weights, col, indptr), shape=(n, n))


def leiden_device(device: str = "auto") -> str:
    """The device :func:`leiden` would run on for ``device``: ``"cpu"`` or ``"cuda:<index>"``.

    ``"auto"`` resolves to ``"cuda:0"`` only when this build of scfast has the ``cugraph``
    feature, ``libcugraph_c.so`` loads (``SCFAST_CUGRAPH_LIB`` overrides the search) and a
    CUDA device is usable; otherwise ``"cpu"``. An explicit ``"cuda"`` / ``"cuda:<index>"``
    that is not usable raises ``RuntimeError`` naming the reason.
    """
    if not isinstance(device, str):
        raise TypeError(f"device must be a string, got {type(device).__name__}")
    return _scfast.leiden_device(device)["resolved"]


def leiden(
    graph: Any,
    resolution: float = 0.5,
    seed: int = 0,
    n_iterations: int = 2,
    device: Literal["auto", "cpu", "cuda"] | str = "auto",
) -> np.ndarray:
    """Leiden clustering (RB / configuration objective), replacing ``FindClusters(algorithm = 4)``.

    Parameters
    ----------
    graph
        Square scipy.sparse CSR matrix (``csr_matrix`` or ``csr_array``): a symmetric
        weighted adjacency with both directions of every edge stored, such as the output
        of :func:`snn`, Seurat's ``RNA_snn`` or Scanpy's ``obsp["connectivities"]``.
        Weights must be finite and non-negative; they are used as float32. A diagonal is
        allowed and treated as Seurat does. Must be canonical (``graph.sum_duplicates()``).
    resolution
        RB resolution (Seurat ``resolution``); larger gives more clusters.
    seed
        Random seed; the same seed and graph give identical labels.
    n_iterations
        Leiden iterations (Seurat ``n.iter``), at least 1. On cuGraph this is ``max_level``.
    device
        ``"auto"`` (cuGraph Leiden when this build has the ``cugraph`` feature and a CUDA
        device is usable, else the CPU), ``"cpu"``, ``"cuda"`` or ``"cuda:<index>"`` (an
        error when cuGraph is not available). See :func:`leiden_device`. Labels follow the
        same numbering on both backends; they are compared by ARI, not expected identical.

    Returns
    -------
    int32 array (cells,)
        Cluster labels 0..c-1, numbered by decreasing cluster size (ties by lowest cell
        index), like Seurat's ``seurat_clusters`` levels.
    """
    if not isinstance(device, str):
        raise TypeError(f"device must be a string, got {type(device).__name__}")
    n_rows, n_cols, indptr, indices, data = _csr_parts(graph, "graph")
    if n_rows != n_cols:
        raise ValueError(f"graph must be square (cells x cells), got shape {(n_rows, n_cols)}")
    if n_rows > _I32_MAX:
        raise ValueError(f"{n_rows} cells exceed the int32 index range")
    resolution = _float("resolution", resolution)
    if resolution < 0:
        raise ValueError(f"resolution must be non-negative, got {resolution}")
    return _scfast.leiden_csr(
        n_rows, indptr, indices, data, resolution, _seed(seed), _int("n_iterations", n_iterations, 1), device
    )


def preprocess(
    X_counts: Any,
    n_hvg: int = 2000,
    scale_factor: float = 1e4,
    span: float = 0.3,
    clip: float | None = None,
) -> dict[str, Any]:
    """Fused log-normalisation and vst variable-gene statistics, replacing Seurat's
    ``NormalizeData`` + ``FindVariableFeatures(selection.method = "vst")`` (Scanpy's
    ``normalize_total`` + ``log1p`` + ``highly_variable_genes(flavor="seurat_v3")`` on the
    same statistics) in one streaming pass over the counts.

    Parameters
    ----------
    X_counts
        scipy.sparse CSR matrix (``csr_matrix`` or ``csr_array``), cells x genes, raw counts.
        float32/float64 data with int32/int64 indices is read in place (zero copy); other
        real dtypes (integer counts included) are converted to float32 first (a copy). Must
        be canonical (``X.sum_duplicates()``). Counts must be finite and non-negative.
    n_hvg
        Number of variable genes (Seurat ``nfeatures``), capped at the number of genes.
    scale_factor
        ``NormalizeData(scale.factor)``: values are ``log1p(x / library_size * scale_factor)``.
    span
        Loess span of the mean-variance trend (Seurat ``loess.span``).
    clip
        Clip for the standardised values (Seurat ``clip.max``); ``None`` means
        ``sqrt(n_cells)``.

    Returns
    -------
    dict
        ``"data"``: scipy.sparse.csr_matrix, float32, cells x genes, the log-normalised
        values (same sparsity pattern as the input, built on its index arrays).
        ``"library_size"``: float64 (cells,). ``"mean"``, ``"variance"``,
        ``"variance_expected"``, ``"variance_standardized"``: float64 (genes,), the vst
        columns (``vst.*`` in Seurat; constant genes get 0 expected and standardised
        variance). ``"hvg"``: int32 (n_hvg,) gene indices, most variable first.
        ``"hvg_mean"``, ``"hvg_sd"``: float64 (n_hvg,), mean and sample standard deviation
        of the log-normalised values of each HVG (what ``ScaleData`` / :func:`pca` scale
        with). ``"clip"``: the clip used. ``"X_hvg"``: ``data[:, hvg]`` as a canonical CSR
        matrix (columns in ``hvg`` order), ready for :func:`pca`.

    Notes
    -----
    The loess is R's ``loess(..., surface = "direct")`` (exact local quadratic fit at
    every gene); Seurat uses R's interpolated surface, so the expected variances differ
    slightly and the HVG sets overlap by > 99% rather than exactly.
    """
    import scipy.sparse as sp

    n_rows, n_cols, indptr, indices, data = _csr_parts(X_counts, "X_counts")
    n_hvg = _int("n_hvg", n_hvg, 1)
    if n_rows == 0 or n_cols == 0:
        raise ValueError(f"X_counts is empty (shape {(n_rows, n_cols)})")
    n_hvg = min(n_hvg, n_cols)
    scale_factor = _float("scale_factor", scale_factor)
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")
    span = _float("span", span)
    if span <= 0:
        raise ValueError(f"span must be positive, got {span}")
    if clip is not None:
        clip = _float("clip", clip)
        if clip <= 0:
            raise ValueError(f"clip must be positive, got {clip}")
    out = _scfast.preprocess_csr(n_rows, n_cols, indptr, indices, data, n_hvg, scale_factor, span, clip)
    norm = sp.csr_matrix((out["data"], indices, indptr), shape=(n_rows, n_cols))
    out["data"] = norm
    x_hvg = norm[:, out["hvg"]]
    x_hvg.sort_indices()  # column selection in HVG order leaves rows unsorted; pca() needs canonical CSR
    out["X_hvg"] = x_hvg
    return out


def _find_ab(spread: float, min_dist: float) -> tuple[float, float]:
    from scipy.optimize import curve_fit

    x = np.linspace(0, spread * 3, 300)
    y = np.where(x < min_dist, 1.0, np.exp(-(x - min_dist) / spread))
    (a, b), _ = curve_fit(lambda x, a, b: 1.0 / (1.0 + a * x ** (2 * b)), x, y)
    return float(a), float(b)


def umap(
    indices: Any,
    distances: Any,
    init: Any,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_epochs: int | None = None,
    negative_sample_rate: int = 5,
    learning_rate: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """UMAP layout from a kNN graph, replacing ``RunUMAP`` / ``sc.tl.umap``.

    ``indices`` / ``distances`` are the (cells, k) output of :func:`knn` (the cell itself
    in column 0). ``init`` is a (cells, 2) starting layout, e.g. the first two PCs; it is
    rescaled to [0, 10] as umap-learn does. ``n_epochs`` defaults to umap-learn's choice
    (500 below 10,000 cells, else 200). ``min_dist`` 0.5 is Scanpy's default. Returns a
    float32 (cells, 2) array.
    """
    idx = np.ascontiguousarray(indices, dtype=np.int32)
    dist = np.ascontiguousarray(distances, dtype=np.float32)
    start = np.ascontiguousarray(np.asarray(init)[:, :2], dtype=np.float32)
    if idx.ndim != 2 or idx.shape != dist.shape or start.shape != (idx.shape[0], 2):
        raise ValueError(f"indices {idx.shape}, distances {dist.shape} and init {start.shape} do not match")
    if n_epochs is None:
        n_epochs = 500 if idx.shape[0] <= 10_000 else 200
    a, b = _find_ab(_float("spread", spread), _float("min_dist", min_dist))
    return _scfast.umap(idx, dist, start, a, b, _int("n_epochs", n_epochs, 1),
                        _int("negative_sample_rate", negative_sample_rate, 0),
                        _float("learning_rate", learning_rate), _seed(seed))
