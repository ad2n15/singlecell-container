//! Python bindings for scfast (docs/RUST_ENGINE.md sections 3 and 5).
//!
//! This crate builds the native extension `scfast._scfast`. The public API (`scfast.pca`,
//! `scfast.knn`, `scfast.snn`, `scfast.leiden`) lives in `python/scfast/__init__.py`, which
//! checks scipy / numpy inputs and calls the low-level functions here. The split keeps all
//! scipy knowledge in Python and all of the following in Rust:
//!
//! - **Zero-copy input**: CSR `indptr` / `indices` / `data` are borrowed straight from the
//!   numpy buffers as a [`CsrView`], dispatching on (int32 | int64) x (float32 | float64).
//!   kNN, SNN and Leiden inputs are copied once, because the core kernels take owned
//!   `RowMajor` / `Knn` / `SparseGraph` values.
//! - **GIL release**: every kernel call (and the O(nnz) validation around it) runs inside
//!   [`Python::detach`], so other Python threads keep running.
//! - **Error mapping**: `Error::InvalidInput` / `InvalidParameter` -> `ValueError`,
//!   `Error::Backend` -> `RuntimeError`. A Rust panic is caught and raised as
//!   `RuntimeError`; `unimplemented!()` (a kernel that is still a stub) is raised as
//!   `NotImplementedError` (a `RuntimeError` subclass), never as a process abort.

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::time::{Duration, Instant};

use numpy::ndarray::Array2;
use numpy::{Element, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use scfast_core::cluster::{self as core_cluster, ClusterParams};
use scfast_core::knn::{self as core_knn, KnnMethod, KnnParams};
use scfast_core::pca::{self as core_pca, PcaParams, PcaResult};
use scfast_core::snn::{self as core_snn, SnnParams};
use scfast_core::{CsrView, Error, Idx, Knn, Real, RowMajor, SparseGraph};

// ------------------------------------------------------------------ errors and panics

/// scfast_core::Error -> Python exception (orphan rule: no `From` impl possible).
fn core_err(e: Error) -> PyErr {
    #[allow(unreachable_patterns)] // stays correct if the core grows new variants
    match e {
        Error::InvalidInput(_) | Error::InvalidParameter(_) => PyValueError::new_err(e.to_string()),
        Error::Backend { .. } => PyRuntimeError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with a non-string payload".to_string()
    }
}

fn panic_err(kernel: &str, payload: Box<dyn Any + Send>) -> PyErr {
    let msg = panic_message(&*payload);
    // `unimplemented!()` panics with "not implemented" / "not implemented: <detail>".
    if msg.starts_with("not implemented") {
        PyNotImplementedError::new_err(format!("scfast.{kernel}: kernel not implemented yet ({msg})"))
    } else {
        PyRuntimeError::new_err(format!("scfast.{kernel}: internal error (Rust panic): {msg}"))
    }
}

/// Runs `f` with the GIL released and converts its errors and panics to Python exceptions.
/// Panics on rayon worker threads are re-raised on this thread by rayon, so they are caught
/// here too (the workspace builds with the default `panic = "unwind"`).
fn run_kernel<T, F>(py: Python<'_>, kernel: &'static str, f: F) -> PyResult<T>
where
    F: FnOnce() -> scfast_core::Result<T> + Send,
    T: Send,
{
    match py.detach(move || panic::catch_unwind(AssertUnwindSafe(f))) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(core_err(e)),
        Err(payload) => Err(panic_err(kernel, payload)),
    }
}

fn backend(context: &'static str, message: String) -> Error {
    Error::Backend { context, message }
}

// ------------------------------------------------------------------ numpy input helpers

fn describe(obj: &Bound<'_, PyAny>) -> String {
    match obj.getattr("dtype").and_then(|d| d.str()) {
        Ok(s) => format!("dtype {s}"),
        Err(_) => match obj.get_type().name() {
            Ok(n) => format!("type {n}"),
            Err(_) => "an unknown type".to_string(),
        },
    }
}

/// Borrowed slice of a C-contiguous numpy array (no copy).
fn contiguous<'a, T: Element, D: numpy::ndarray::Dimension>(a: &'a PyReadonlyArray<'_, T, D>, name: &str) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

/// CSR index arrays; `indptr` and `indices` share one dtype, as scipy guarantees.
enum IndexArrays<'py> {
    I32(PyReadonlyArray1<'py, i32>, PyReadonlyArray1<'py, i32>),
    I64(PyReadonlyArray1<'py, i64>, PyReadonlyArray1<'py, i64>),
}

impl<'py> IndexArrays<'py> {
    fn extract(indptr: &Bound<'py, PyAny>, indices: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let (Ok(p), Ok(i)) = (indptr.extract::<PyReadonlyArray1<i32>>(), indices.extract::<PyReadonlyArray1<i32>>()) {
            return Ok(Self::I32(p, i));
        }
        if let (Ok(p), Ok(i)) = (indptr.extract::<PyReadonlyArray1<i64>>(), indices.extract::<PyReadonlyArray1<i64>>()) {
            return Ok(Self::I64(p, i));
        }
        Err(PyTypeError::new_err(format!(
            "indptr and indices must be 1-D numpy arrays sharing one dtype, int32 or int64; got {} and {}",
            describe(indptr),
            describe(indices)
        )))
    }
}

enum ValueArray<'py> {
    F32(PyReadonlyArray1<'py, f32>),
    F64(PyReadonlyArray1<'py, f64>),
}

impl<'py> ValueArray<'py> {
    fn extract(data: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(d) = data.extract::<PyReadonlyArray1<f32>>() {
            return Ok(Self::F32(d));
        }
        if let Ok(d) = data.extract::<PyReadonlyArray1<f64>>() {
            return Ok(Self::F64(d));
        }
        Err(PyTypeError::new_err(format!("data must be a 1-D numpy array of float32 or float64; got {}", describe(data))))
    }
}

/// Expands `$body` once per (index, value) dtype pair with `$ip`, `$ix`, `$dx` bound to the
/// borrowed `&[I]`, `&[I]`, `&[T]` slices, so the body is monomorphised for each pair.
macro_rules! dispatch_csr {
    ($ind:expr, $val:expr, |$ip:ident, $ix:ident, $dx:ident| $body:expr) => {
        match (&$ind, &$val) {
            (IndexArrays::I32(p, i), ValueArray::F32(d)) => {
                let ($ip, $ix, $dx) = (contiguous(p, "indptr")?, contiguous(i, "indices")?, contiguous(d, "data")?);
                $body
            }
            (IndexArrays::I32(p, i), ValueArray::F64(d)) => {
                let ($ip, $ix, $dx) = (contiguous(p, "indptr")?, contiguous(i, "indices")?, contiguous(d, "data")?);
                $body
            }
            (IndexArrays::I64(p, i), ValueArray::F32(d)) => {
                let ($ip, $ix, $dx) = (contiguous(p, "indptr")?, contiguous(i, "indices")?, contiguous(d, "data")?);
                $body
            }
            (IndexArrays::I64(p, i), ValueArray::F64(d)) => {
                let ($ip, $ix, $dx) = (contiguous(p, "indptr")?, contiguous(i, "indices")?, contiguous(d, "data")?);
                $body
            }
        }
    };
}

/// Canonical CSR: column indices strictly increasing within each row (sorted, no
/// duplicates). `CsrView::new` checks structure and bounds but not this; duplicates would
/// break the per-gene variance in PCA and the symmetry check in Leiden.
fn check_canonical<I: Idx, T: Real>(x: &CsrView<'_, I, T>, what: &str) -> scfast_core::Result<()> {
    let bad = (0..x.n_rows).into_par_iter().find_first(|&r| {
        let (cols, _) = x.row(r);
        cols.windows(2).any(|w| w[1].to_usize() <= w[0].to_usize())
    });
    match bad {
        None => Ok(()),
        Some(r) => Err(Error::InvalidInput(format!(
            "{what} is not in canonical CSR format: row {r} has unsorted or duplicate column indices \
             (call {what}.sum_duplicates() first; it also sorts the indices)"
        ))),
    }
}

fn type_tag<T: 'static>(_: &[T]) -> &'static str {
    std::any::type_name::<T>()
}

// ------------------------------------------------------------------ numpy output helpers

fn rowmajor_to_py<'py>(py: Python<'py>, m: RowMajor, what: &str) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (r, c) = (m.n_rows, m.n_cols);
    let a = Array2::from_shape_vec((r, c), m.data)
        .map_err(|e| PyRuntimeError::new_err(format!("scfast: {what} has an inconsistent shape {r}x{c}: {e}")))?;
    Ok(a.into_pyarray(py))
}

fn vec2_to_py<'py, T: Element>(py: Python<'py>, n: usize, k: usize, v: Vec<T>) -> PyResult<Bound<'py, PyArray2<T>>> {
    let a = Array2::from_shape_vec((n, k), v).map_err(|e| PyRuntimeError::new_err(format!("scfast: bad output shape: {e}")))?;
    Ok(a.into_pyarray(py))
}

fn to_i32(v: Vec<u32>) -> Vec<i32> {
    // Callers guarantee every value is < 2^31 (n <= i32::MAX); same layout, collected in place.
    v.into_iter().map(|x| x as i32).collect()
}

fn check_n_fits_i32(n: usize, what: &str) -> PyResult<()> {
    if n > i32::MAX as usize {
        return Err(PyValueError::new_err(format!("{what}: {n} cells exceed the int32 index range")));
    }
    Ok(())
}

// ------------------------------------------------------------------ PCA

/// Low-level PCA on borrowed CSR arrays (cells x genes). See scfast.pca.
#[pyfunction]
#[pyo3(name = "pca_csr", signature = (n_rows, n_cols, indptr, indices, data, n_comps, seed, scale, oversample, power_iters))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn pca_csr<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
    n_comps: usize,
    seed: u64,
    scale: bool,
    oversample: usize,
    power_iters: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>)> {
    let params = PcaParams { n_comps, oversample, power_iters, seed, scale, ..PcaParams::default() };
    let ind = IndexArrays::extract(indptr, indices)?;
    let val = ValueArray::extract(data)?;
    let res: PcaResult = dispatch_csr!(ind, val, |ip, ix, dx| run_kernel(py, "pca", || {
        let x = CsrView::new(n_rows, n_cols, ip, ix, dx)?;
        check_canonical(&x, "X")?;
        let r = core_pca::implicit_scaled_pca(x, &params)?;
        let k = r.stdev.len();
        if r.embeddings.n_rows != n_rows || r.loadings.n_rows != n_cols || r.embeddings.n_cols != k || r.loadings.n_cols != k {
            return Err(backend("pca", format!(
                "kernel returned embeddings {}x{}, loadings {}x{}, {} stdevs for a {n_rows}x{n_cols} input",
                r.embeddings.n_rows, r.embeddings.n_cols, r.loadings.n_rows, r.loadings.n_cols, k
            )));
        }
        Ok(r)
    }))?;
    Ok((
        rowmajor_to_py(py, res.embeddings, "embeddings")?,
        rowmajor_to_py(py, res.loadings, "loadings")?,
        PyArray1::from_vec(py, res.stdev),
    ))
}

/// Diagnostics for tests: runs the same dtype dispatch and validation as pca_csr and reports
/// which Rust types were chosen and where the borrowed slices point (zero-copy check).
#[pyfunction]
#[pyo3(name = "csr_view_info", signature = (n_rows, n_cols, indptr, indices, data))]
fn csr_view_info<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let ind = IndexArrays::extract(indptr, indices)?;
    let val = ValueArray::extract(data)?;
    let (ity, vty, pp, ipp, dp, nnz) = dispatch_csr!(ind, val, |ip, ix, dx| run_kernel(py, "csr_view_info", || {
        let x = CsrView::new(n_rows, n_cols, ip, ix, dx)?;
        check_canonical(&x, "X")?;
        Ok((type_tag(x.indptr), type_tag(x.data), x.indptr.as_ptr() as usize, x.indices.as_ptr() as usize, x.data.as_ptr() as usize, x.nnz()))
    }))?;
    let d = PyDict::new(py);
    d.set_item("index_type", ity)?;
    d.set_item("value_type", vty)?;
    d.set_item("indptr_ptr", pp)?;
    d.set_item("indices_ptr", ipp)?;
    d.set_item("data_ptr", dp)?;
    d.set_item("n_rows", n_rows)?;
    d.set_item("n_cols", n_cols)?;
    d.set_item("nnz", nnz)?;
    Ok(d)
}

// ------------------------------------------------------------------ kNN

/// Low-level kNN on a C-contiguous float32 (n, d) array. See scfast.knn.
#[pyfunction]
#[pyo3(name = "knn", signature = (x, k, method, seed, auto_threshold, m, ef_construction, ef_search))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn knn_dense<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    k: usize,
    method: &str,
    seed: u64,
    auto_threshold: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<f32>>)> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    check_n_fits_i32(n, "knn")?;
    let hnsw = KnnMethod::Hnsw { m, ef_construction, ef_search };
    // "auto" is resolved here rather than with KnnMethod::Auto so that the caller's HNSW
    // parameters apply in auto mode too (the core's Auto uses fixed ones).
    let method = match method {
        "exact" => KnnMethod::Exact,
        "hnsw" => hnsw,
        "auto" if n < auto_threshold => KnnMethod::Exact,
        "auto" => hnsw,
        other => return Err(PyValueError::new_err(format!("method must be 'auto', 'exact' or 'hnsw', got {other:?}"))),
    };
    let xs = contiguous(&x, "X")?;
    let params = KnnParams { k, method, auto_threshold, seed };
    let (idx, dist) = run_kernel(py, "knn", || {
        if let Some(p) = xs.iter().position(|v| !v.is_finite()) {
            return Err(Error::InvalidInput(format!("X has a non-finite value at row {}, column {}", p / d.max(1), p % d.max(1))));
        }
        let rm = RowMajor::from_vec(n, d, xs.to_vec())?;
        let r = core_knn::knn(&rm, &params)?;
        if r.n != n || r.k != k || r.indices.len() != n * k || r.distances.len() != n * k {
            return Err(backend("knn", format!("kernel returned n={}, k={} ({} indices, {} distances) for n={n}, k={k}", r.n, r.k, r.indices.len(), r.distances.len())));
        }
        Ok((to_i32(r.indices), r.distances))
    })?;
    Ok((vec2_to_py(py, n, k, idx)?, vec2_to_py(py, n, k, dist)?))
}

// ------------------------------------------------------------------ SNN

enum KnnIndexArray<'py> {
    I32(PyReadonlyArray2<'py, i32>),
    I64(PyReadonlyArray2<'py, i64>),
    U32(PyReadonlyArray2<'py, u32>),
    U64(PyReadonlyArray2<'py, u64>),
}

/// Copies kNN indices into u32, checking every value is in [0, n).
fn knn_indices_to_u32<T>(s: &[T], n: usize, k: usize) -> scfast_core::Result<Vec<u32>>
where
    T: Copy + Send + Sync + std::fmt::Display,
    u32: TryFrom<T>,
{
    let mut out = Vec::with_capacity(s.len());
    for (p, &v) in s.iter().enumerate() {
        match u32::try_from(v) {
            Ok(u) if (u as usize) < n => out.push(u),
            _ => {
                return Err(Error::InvalidInput(format!(
                    "indices[{}, {}] = {v} is not a valid cell index for {n} cells",
                    p / k,
                    p % k
                )))
            }
        }
    }
    Ok(out)
}

/// Low-level SNN from (n, k) kNN indices. Returns (indptr, indices, weights) of the full
/// symmetric CSR adjacency; indptr is int32 when nnz fits, else int64.
#[pyfunction]
#[pyo3(name = "snn", signature = (indices, prune, keep_diagonal))]
#[allow(clippy::type_complexity)]
fn snn_from_indices<'py>(
    py: Python<'py>,
    indices: &Bound<'py, PyAny>,
    prune: f64,
    keep_diagonal: bool,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f32>>)> {
    let arr = if let Ok(a) = indices.extract::<PyReadonlyArray2<i32>>() {
        KnnIndexArray::I32(a)
    } else if let Ok(a) = indices.extract::<PyReadonlyArray2<i64>>() {
        KnnIndexArray::I64(a)
    } else if let Ok(a) = indices.extract::<PyReadonlyArray2<u32>>() {
        KnnIndexArray::U32(a)
    } else if let Ok(a) = indices.extract::<PyReadonlyArray2<u64>>() {
        KnnIndexArray::U64(a)
    } else {
        return Err(PyTypeError::new_err(format!(
            "indices must be a 2-D numpy array of int32, int64, uint32 or uint64; got {}",
            describe(indices)
        )));
    };
    let shape = indices.getattr("shape")?.extract::<(usize, usize)>()?;
    let (n, k) = shape;
    check_n_fits_i32(n, "snn")?;
    let params = SnnParams { prune, keep_diagonal };
    macro_rules! go {
        ($a:expr) => {{
            let s = contiguous($a, "indices")?;
            run_kernel(py, "snn", || {
                let idx = knn_indices_to_u32(s, n, k.max(1))?;
                let knn = Knn { n, k, indices: idx, distances: vec![0.0; n * k] };
                core_snn::snn_from_knn(&knn, &params)
            })?
        }};
    }
    let g: SparseGraph = match &arr {
        KnnIndexArray::I32(a) => go!(a),
        KnnIndexArray::I64(a) => go!(a),
        KnnIndexArray::U32(a) => go!(a),
        KnnIndexArray::U64(a) => go!(a),
    };
    let nnz = g.nnz();
    if g.n != n || g.indptr.len() != n + 1 || g.weights.len() != nnz || g.indptr[n] != nnz {
        return Err(PyRuntimeError::new_err(format!(
            "scfast.snn: kernel returned an inconsistent graph (n={}, {} indptr entries, {} indices, {} weights) for n={n}",
            g.n,
            g.indptr.len(),
            nnz,
            g.weights.len()
        )));
    }
    let indptr: Bound<'py, PyAny> = if nnz <= i32::MAX as usize {
        PyArray1::from_vec(py, g.indptr.into_iter().map(|v| v as i32).collect::<Vec<i32>>()).into_any()
    } else {
        PyArray1::from_vec(py, g.indptr.into_iter().map(|v| v as i64).collect::<Vec<i64>>()).into_any()
    };
    Ok((indptr, PyArray1::from_vec(py, to_i32(g.indices)), PyArray1::from_vec(py, g.weights)))
}

// ------------------------------------------------------------------ Leiden

/// Owned copy of a validated square CSR as the core's SparseGraph (the core takes owned
/// buffers). Symmetry, weight sign and finiteness are checked by the core (single-clustering).
fn graph_from_csr<I: Idx, T: Real>(x: &CsrView<'_, I, T>) -> SparseGraph {
    SparseGraph {
        n: x.n_rows,
        indptr: x.indptr.iter().map(|v| v.to_usize()).collect(),
        indices: x.indices.iter().map(|v| v.to_usize() as u32).collect(),
        weights: x.data.iter().map(|v| v.to_f32()).collect(),
    }
}

/// Low-level Leiden on a square, symmetric, canonical CSR graph. Returns int32 labels.
#[pyfunction]
#[pyo3(name = "leiden_csr", signature = (n, indptr, indices, data, resolution, seed, n_iterations))]
#[allow(clippy::too_many_arguments)]
fn leiden_csr<'py>(
    py: Python<'py>,
    n: usize,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
    resolution: f64,
    seed: u64,
    n_iterations: usize,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    check_n_fits_i32(n, "leiden")?;
    let params = ClusterParams { resolution, seed, n_iterations };
    let ind = IndexArrays::extract(indptr, indices)?;
    let val = ValueArray::extract(data)?;
    let labels: Vec<u32> = dispatch_csr!(ind, val, |ip, ix, dx| run_kernel(py, "leiden", || {
        let x = CsrView::new(n, n, ip, ix, dx)?;
        check_canonical(&x, "graph")?;
        let g = graph_from_csr(&x);
        let labels = core_cluster::leiden(&g, &params)?;
        if labels.len() != n {
            return Err(backend("leiden", format!("kernel returned {} labels for {n} nodes", labels.len())));
        }
        Ok(labels)
    }))?;
    Ok(PyArray1::from_vec(py, to_i32(labels)))
}

// ------------------------------------------------------------------ test probes

/// Busy-spins for `seconds` inside the same GIL-releasing wrapper the kernels use.
/// Returns the measured duration. For tests of GIL release that do not need a kernel.
#[pyfunction]
#[pyo3(name = "_probe_detach")]
fn probe_detach(py: Python<'_>, seconds: f64) -> PyResult<f64> {
    let dur = Duration::from_secs_f64(seconds.max(0.0));
    run_kernel(py, "_probe_detach", move || {
        let t = Instant::now();
        let mut acc = 0u64;
        while t.elapsed() < dur {
            acc = std::hint::black_box(acc.wrapping_add(1));
        }
        Ok(t.elapsed().as_secs_f64())
    })
}

/// Raises the Python exception that a scfast_core::Error of `kind` maps to
/// ("invalid_input", "invalid_parameter" or "backend").
#[pyfunction]
#[pyo3(name = "_probe_error")]
fn probe_error(py: Python<'_>, kind: &str, message: String) -> PyResult<()> {
    let e = match kind {
        "invalid_input" => Error::InvalidInput(message),
        "invalid_parameter" => Error::InvalidParameter(message),
        "backend" => backend("probe", message),
        other => return Err(PyValueError::new_err(format!("unknown error kind {other:?}"))),
    };
    run_kernel(py, "_probe_error", move || Err(e))
}

/// Panics inside the kernel wrapper: kind "panic" (plain panic), "unimplemented" (like a
/// stub kernel) or "rayon" (panic on a rayon worker thread).
#[pyfunction]
#[pyo3(name = "_probe_panic")]
fn probe_panic(py: Python<'_>, kind: &str) -> PyResult<()> {
    let kind = kind.to_string();
    run_kernel(py, "_probe_panic", move || -> scfast_core::Result<()> {
        match kind.as_str() {
            "panic" => panic!("probe panic"),
            "unimplemented" => unimplemented!("probe stub"),
            "rayon" => {
                (0..1024usize).into_par_iter().for_each(|i| {
                    if i == 777 {
                        panic!("probe panic on a rayon worker");
                    }
                });
                Ok(())
            }
            other => Err(Error::InvalidParameter(format!("unknown panic kind {other:?}"))),
        }
    })
}

// ------------------------------------------------------------------ module

/// Native core of scfast (Rust). Use the `scfast` package, not this module, directly.
#[pymodule(name = "_scfast")]
mod native {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::{csr_view_info, knn_dense, leiden_csr, pca_csr, probe_detach, probe_error, probe_panic, snn_from_indices};

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))
    }
}
