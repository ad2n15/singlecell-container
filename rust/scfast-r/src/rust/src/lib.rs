//! extendr bindings for scfast-core: the native side of the R package scfastR.
//!
//! Each `rs_*` function is a thin adapter: it borrows R memory, calls one scfast-core kernel
//! and converts the result to plain R vectors / matrices / lists. All Seurat-specific logic
//! (DimReduc, Graph, meta data) lives in R (R/scfast.R).
//!
//! Copies (docs/RUST_ENGINE.md section 3):
//! - PCA input: the dgCMatrix `@p`, `@i`, `@x` arrays are borrowed in place as a cells x genes
//!   `CsrView<i32, f64>`. No copy, no transpose.
//! - kNN input: the R embedding matrix (column-major f64) is borrowed, then written once into
//!   the row-major f32 `RowMajor` that `knn::knn` takes (cells x dims, small).
//! - SNN input: the 1-based integer index matrix is converted to the 0-based row-major `u32`
//!   `Knn` (n x k, small).
//! - Leiden input: the dgCMatrix arrays are converted to the owned `usize`/`u32`/`f32`
//!   `SparseGraph` that the clusterer takes (one copy of the graph, as planned in section 4).
//! - Outputs are always new R vectors (f32 -> f64, row-major -> column-major).
//!
//! Errors: scfast-core `Error`s and Rust panics (including `unimplemented!()` in stub
//! kernels) are caught here and returned as `Err`, which the extendr wrapper turns into an
//! R error. Nothing unwinds into R.

use extendr_api::prelude::*;
use scfast_core::{cluster, knn, pca, snn, CsrView, Knn, RowMajor, SparseGraph};
use std::panic::{self, AssertUnwindSafe};
use std::sync::Once;

type RResult<T> = std::result::Result<T, extendr_api::Error>;

fn err<T>(msg: impl Into<String>) -> RResult<T> {
    Err(extendr_api::Error::Other(msg.into()))
}

// ------------------------------------------------------------------ panic handling

static QUIET_HOOK: Once = Once::new();

/// Replace the default panic hook (which prints "thread panicked at ..." to stderr) with a
/// silent one, because every panic is reported as an R error instead. Set the environment
/// variable SCFAST_DEBUG=1 to keep the default hook (and RUST_BACKTRACE to get backtraces).
fn install_quiet_hook() {
    QUIET_HOOK.call_once(|| {
        if std::env::var_os("SCFAST_DEBUG").is_none() {
            panic::set_hook(Box::new(|_| {}));
        }
    });
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Run one kernel, turning a scfast-core error or a panic into an R-facing error message
/// that names the kernel. Stub kernels (`unimplemented!()`) produce
/// "scfast <kernel>: not implemented: ...", which the R side maps to the condition class
/// `scfast_not_implemented`.
fn run_kernel<T>(kernel: &str, f: impl FnOnce() -> scfast_core::Result<T>) -> RResult<T> {
    install_quiet_hook();
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => err(format!("scfast {kernel}: {e}")),
        Err(payload) => err(format!("scfast {kernel}: {}", panic_message(payload.as_ref()))),
    }
}

// ------------------------------------------------------------------ argument helpers

fn int_slice<'a>(x: &'a Robj, what: &str) -> RResult<&'a [i32]> {
    match x.as_integer_slice() {
        Some(s) => Ok(s),
        None => err(format!("{what} must be an integer vector, got {:?}", x.rtype())),
    }
}

fn real_slice<'a>(x: &'a Robj, what: &str) -> RResult<&'a [f64]> {
    match x.as_real_slice() {
        Some(s) => Ok(s),
        None => err(format!("{what} must be a double vector, got {:?}", x.rtype())),
    }
}

fn non_negative(v: i32, what: &str) -> RResult<usize> {
    if v < 0 {
        return err(format!("{what} must be >= 0, got {v}"));
    }
    Ok(v as usize)
}

fn seed_u64(seed: f64) -> RResult<u64> {
    if !seed.is_finite() || seed < 0.0 || seed.fract() != 0.0 || seed > 9.007_199_254_740_992e15 {
        return err(format!("seed must be a non-negative whole number, got {seed}"));
    }
    Ok(seed as u64)
}

/// Borrow a genes x cells dgCMatrix (`@p`, `@i`, `@x`) as a cells x genes CSR view, in
/// place. Index validation (O(nnz)) is done by `CsrView::new`; values are checked for
/// NA/NaN/Inf here.
fn csc_as_csr<'a>(p: &'a Robj, i: &'a Robj, x: &'a Robj, n_genes: i32, n_cells: i32) -> RResult<CsrView<'a, i32, f64>> {
    let n_genes = non_negative(n_genes, "number of genes")?;
    let n_cells = non_negative(n_cells, "number of cells")?;
    let p = int_slice(p, "@p")?;
    let i = int_slice(i, "@i")?;
    let x = real_slice(x, "@x")?;
    if p.iter().any(|&v| v < 0) || i.iter().any(|&v| v < 0) {
        return err("invalid input: negative (or NA) entries in @p or @i");
    }
    if let Some(pos) = x.iter().position(|v| !v.is_finite()) {
        return err(format!("invalid input: @x contains a non-finite value (NA, NaN or Inf) at position {}", pos + 1));
    }
    CsrView::new(n_cells, n_genes, p, i, x).map_err(|e| extendr_api::Error::Other(format!("scfast: {e}")))
}

/// Row-major f32 -> new R double matrix (column-major).
fn rowmajor_to_rmatrix(m: &RowMajor) -> Robj {
    let (nr, nc) = (m.n_rows, m.n_cols);
    let mut out = RMatrix::<f64>::new(nr, nc);
    let dst = out.data_mut();
    for c in 0..nc {
        let col = &mut dst[c * nr..(c + 1) * nr];
        for (r, v) in col.iter_mut().enumerate() {
            *v = m.data[r * nc + c] as f64;
        }
    }
    out.into()
}

fn f32_to_r(v: &[f32]) -> Robj {
    v.iter().map(|&x| x as f64).collect::<Vec<f64>>().into()
}

// ------------------------------------------------------------------ PCA

/// Implicitly scaled PCA of a genes x cells dgCMatrix, read in place as cells x genes CSR.
/// `oversample` / `power_iters` < 0 keep the scfast-core defaults.
/// @keywords internal
#[extendr]
fn rs_pca_csc(p: Robj, i: Robj, x: Robj, n_genes: i32, n_cells: i32, npcs: i32, seed: f64, oversample: i32, power_iters: i32) -> RResult<Robj> {
    let view = csc_as_csr(&p, &i, &x, n_genes, n_cells)?;
    if npcs < 1 {
        return err(format!("npcs must be >= 1, got {npcs}"));
    }
    let mut params = pca::PcaParams { n_comps: npcs as usize, seed: seed_u64(seed)?, ..Default::default() };
    if oversample >= 0 {
        params.oversample = oversample as usize;
    }
    if power_iters >= 0 {
        params.power_iters = power_iters as usize;
    }
    let res = run_kernel("pca", || pca::implicit_scaled_pca(view, &params))?;
    if res.embeddings.n_rows != view.n_rows || res.loadings.n_rows != view.n_cols {
        return err(format!(
            "scfast pca: kernel returned embeddings {}x{} and loadings {}x{} for {} cells x {} genes",
            res.embeddings.n_rows, res.embeddings.n_cols, res.loadings.n_rows, res.loadings.n_cols, view.n_rows, view.n_cols
        ));
    }
    Ok(list!(
        embeddings = rowmajor_to_rmatrix(&res.embeddings),
        loadings = rowmajor_to_rmatrix(&res.loadings),
        stdev = f32_to_r(&res.stdev),
        gene_mean = f32_to_r(&res.gene_mean),
        gene_sd = f32_to_r(&res.gene_sd)
    )
    .into())
}

// ------------------------------------------------------------------ kNN

fn knn_method(method: &str, hnsw_m: i32, ef_construction: i32, ef_search: i32) -> RResult<knn::KnnMethod> {
    match method {
        "auto" => Ok(knn::KnnMethod::Auto),
        "exact" => Ok(knn::KnnMethod::Exact),
        "hnsw" => {
            if hnsw_m < 1 || ef_construction < 1 || ef_search < 1 {
                return err("hnsw parameters (m, ef_construction, ef_search) must be >= 1");
            }
            Ok(knn::KnnMethod::Hnsw { m: hnsw_m as usize, ef_construction: ef_construction as usize, ef_search: ef_search as usize })
        }
        other => err(format!("method must be one of \"auto\", \"exact\", \"hnsw\"; got \"{other}\"")),
    }
}

/// Exact / HNSW k nearest neighbours (self included) of every row of a double matrix
/// (cells x dims). Returns 1-based indices and Euclidean distances, both n x k.
/// `auto_threshold` < 0 keeps the scfast-core default.
/// @keywords internal
#[extendr]
fn rs_knn(mat: Robj, n: i32, d: i32, k: i32, method: &str, seed: f64, hnsw_m: i32, ef_construction: i32, ef_search: i32, auto_threshold: f64) -> RResult<Robj> {
    let n = non_negative(n, "number of cells")?;
    let d = non_negative(d, "number of dimensions")?;
    let x = real_slice(&mat, "embedding matrix")?;
    if x.len() != n * d {
        return err(format!("embedding matrix has {} values, expected {n} x {d}", x.len()));
    }
    if let Some(pos) = x.iter().position(|v| !v.is_finite()) {
        return err(format!("embedding matrix contains a non-finite value (NA, NaN or Inf) at position {}", pos + 1));
    }
    if k < 2 || k as usize > n {
        return err(format!("k must be between 2 (it counts the cell itself) and the number of cells ({n}), got {k}"));
    }
    let mut params = knn::KnnParams { k: k as usize, method: knn_method(method, hnsw_m, ef_construction, ef_search)?, seed: seed_u64(seed)?, ..Default::default() };
    if auto_threshold >= 0.0 {
        params.auto_threshold = auto_threshold as usize;
    }
    // Column-major f64 (R) -> row-major f32 (scfast). The only copy, n x d.
    let mut rm = vec![0f32; n * d];
    for c in 0..d {
        let col = &x[c * n..(c + 1) * n];
        for (r, &v) in col.iter().enumerate() {
            rm[r * d + c] = v as f32;
        }
    }
    let rm = RowMajor::from_vec(n, d, rm).map_err(|e| extendr_api::Error::Other(format!("scfast: {e}")))?;
    let res = run_kernel("knn", || knn::knn(&rm, &params))?;
    let kk = res.k;
    if res.n != n || res.indices.len() != n * kk || res.distances.len() != n * kk {
        return err(format!("scfast knn: kernel returned n = {}, k = {kk} with {} indices for {n} cells", res.n, res.indices.len()));
    }
    let mut idx = RMatrix::<i32>::new(n, kk);
    let mut dist = RMatrix::<f64>::new(n, kk);
    {
        let di = idx.data_mut();
        for c in 0..kk {
            for r in 0..n {
                di[c * n + r] = res.indices[r * kk + c] as i32 + 1;
            }
        }
    }
    {
        let dd = dist.data_mut();
        for c in 0..kk {
            for r in 0..n {
                dd[c * n + r] = res.distances[r * kk + c] as f64;
            }
        }
    }
    Ok(list!(idx = idx, dist = dist).into())
}

// ------------------------------------------------------------------ SNN

fn graph_to_r(g: &SparseGraph) -> RResult<Robj> {
    if g.nnz() > i32::MAX as usize {
        return err(format!("graph has {} edges, more than an R dgCMatrix can hold", g.nnz()));
    }
    if g.indptr.len() != g.n + 1 || g.weights.len() != g.indices.len() {
        return err("scfast: kernel returned an inconsistent graph");
    }
    let p: Vec<i32> = g.indptr.iter().map(|&v| v as i32).collect();
    let i: Vec<i32> = g.indices.iter().map(|&v| v as i32).collect();
    let x: Vec<f64> = g.weights.iter().map(|&v| v as f64).collect();
    Ok(list!(p = p, i = i, x = x, n = g.n as i32).into())
}

/// SNN (Jaccard) graph from an n x k matrix of 1-based neighbour indices (self first).
/// Returns the full symmetric adjacency as CSR parts (0-based), which are also its CSC parts.
/// @keywords internal
#[extendr]
fn rs_snn(idx: Robj, n: i32, k: i32, prune: f64) -> RResult<Robj> {
    let n = non_negative(n, "number of cells")?;
    let k = non_negative(k, "k")?;
    let ix = int_slice(&idx, "neighbour index matrix")?;
    if ix.len() != n * k || k == 0 {
        return err(format!("neighbour index matrix has {} values, expected {n} x {k} (k >= 1)", ix.len()));
    }
    if !prune.is_finite() || !(0.0..=1.0).contains(&prune) {
        return err(format!("prune must be in [0, 1], got {prune}"));
    }
    let mut indices = vec![0u32; n * k];
    for c in 0..k {
        for r in 0..n {
            let v = ix[c * n + r];
            if v < 1 || v as usize > n {
                return err(format!("neighbour index {v} (row {}, column {}) is outside 1..{n}", r + 1, c + 1));
            }
            indices[r * k + c] = (v - 1) as u32;
        }
    }
    // snn_from_knn reads only the indices; distances may be empty.
    let knn = Knn { n, k, indices, distances: Vec::new() };
    let params = snn::SnnParams { prune, ..Default::default() };
    let g = run_kernel("snn", || snn::snn_from_knn(&knn, &params))?;
    if g.n != n {
        return err(format!("scfast snn: kernel returned a graph of {} nodes for {n} cells", g.n));
    }
    graph_to_r(&g)
}

// ------------------------------------------------------------------ Leiden

/// Leiden (RB objective) on a symmetric n x n dgCMatrix given by its `@p`, `@i`, `@x`.
/// Returns 0-based labels numbered by decreasing cluster size.
/// @keywords internal
#[extendr]
fn rs_leiden(p: Robj, i: Robj, x: Robj, n: i32, resolution: f64, seed: f64, n_iter: i32) -> RResult<Robj> {
    let n = non_negative(n, "number of nodes")?;
    let p = int_slice(&p, "@p")?;
    let i = int_slice(&i, "@i")?;
    let x = real_slice(&x, "@x")?;
    if p.len() != n + 1 || i.len() != x.len() || p.first() != Some(&0) || p[n] as usize != x.len() {
        return err("invalid input: graph arrays are inconsistent (need length(p) = n + 1, p[1] = 0, p[n + 1] = length(i) = length(x))");
    }
    if p.windows(2).any(|w| w[1] < w[0]) {
        return err("invalid input: graph @p is not non-decreasing");
    }
    if i.iter().any(|&v| v < 0 || v as usize >= n) {
        return err(format!("invalid input: graph @i has entries outside 0..{}", n.saturating_sub(1)));
    }
    if x.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return err("invalid input: graph weights must be finite and non-negative");
    }
    if !resolution.is_finite() || resolution < 0.0 {
        return err(format!("resolution must be >= 0, got {resolution}"));
    }
    if n_iter < 1 {
        return err(format!("n_iter must be >= 1, got {n_iter}"));
    }
    let graph = SparseGraph {
        n,
        indptr: p.iter().map(|&v| v as usize).collect(),
        indices: i.iter().map(|&v| v as u32).collect(),
        weights: x.iter().map(|&v| v as f32).collect(),
    };
    let params = cluster::ClusterParams { resolution, seed: seed_u64(seed)?, n_iterations: n_iter as usize };
    let labels = run_kernel("leiden", || cluster::leiden(&graph, &params))?;
    if labels.len() != n {
        return err(format!("scfast leiden: kernel returned {} labels for {n} nodes", labels.len()));
    }
    Ok(labels.iter().map(|&l| l as i32).collect::<Vec<i32>>().into())
}

/// Is the n x n dgCMatrix (`@p`, `@i`, `@x`, row indices sorted within columns, as in any
/// valid dgCMatrix) symmetric to a relative tolerance? O(nnz log degree), no allocation,
/// unlike `isSymmetric()` in R, which builds the transpose.
/// @keywords internal
#[extendr]
fn rs_is_symmetric(p: Robj, i: Robj, x: Robj, n: i32, tol: f64) -> RResult<bool> {
    let n = non_negative(n, "number of nodes")?;
    let p = int_slice(&p, "@p")?;
    let i = int_slice(&i, "@i")?;
    let x = real_slice(&x, "@x")?;
    if p.len() != n + 1 || i.len() != x.len() || p[n] as usize != x.len() {
        return err("invalid input: graph arrays are inconsistent");
    }
    for col in 0..n {
        let (s, e) = (p[col] as usize, p[col + 1] as usize);
        for k in s..e {
            let row = i[k];
            if row < 0 || row as usize >= n {
                return err("invalid input: graph @i out of range");
            }
            let row = row as usize;
            if row == col {
                continue;
            }
            // Look for (col, row), i.e. row index `col` inside column `row`.
            let (s2, e2) = (p[row] as usize, p[row + 1] as usize);
            match i[s2..e2].binary_search(&(col as i32)) {
                Ok(pos) => {
                    let (a, b) = (x[k], x[s2 + pos]);
                    if (a - b).abs() > tol * a.abs().max(b.abs()).max(f64::MIN_POSITIVE) {
                        return Ok(false);
                    }
                }
                Err(_) => {
                    if x[k] != 0.0 {
                        return Ok(false);
                    }
                }
            }
        }
    }
    Ok(true)
}

// ------------------------------------------------------------------ diagnostics

/// Build information: crate version, rayon thread count, compiled backends.
/// @keywords internal
#[extendr]
fn rs_info() -> Robj {
    list!(
        version = env!("CARGO_PKG_VERSION"),
        threads = rayon::current_num_threads() as i32,
        cuda = false,
        extendr = "0.9"
    )
    .into()
}

/// Address of an R vector's data, as a hex string (zero-copy checks).
/// @keywords internal
#[extendr]
fn rs_dataptr(x: Robj) -> RResult<String> {
    let addr = if let Some(s) = x.as_real_slice() {
        s.as_ptr() as usize
    } else if let Some(s) = x.as_integer_slice() {
        s.as_ptr() as usize
    } else {
        return err("rs_dataptr: x must be a double or integer vector");
    };
    Ok(format!("{addr:#x}"))
}

/// The data addresses inside the CsrView that `rs_pca_csc` builds from the same arguments
/// (zero-copy checks: they must equal `rs_dataptr()` of `@p`, `@i`, `@x`).
/// @keywords internal
#[extendr]
fn rs_csr_view_ptrs(p: Robj, i: Robj, x: Robj, n_genes: i32, n_cells: i32) -> RResult<Robj> {
    let v = csc_as_csr(&p, &i, &x, n_genes, n_cells)?;
    Ok(list!(
        indptr = format!("{:#x}", v.indptr.as_ptr() as usize),
        indices = format!("{:#x}", v.indices.as_ptr() as usize),
        data = format!("{:#x}", v.data.as_ptr() as usize),
        n_rows = v.n_rows as i32,
        n_cols = v.n_cols as i32,
        nnz = v.nnz() as i32
    )
    .into())
}

/// Panics on purpose (checks that panics become R errors).
/// @keywords internal
#[extendr]
fn rs_debug_panic(msg: &str) -> RResult<Robj> {
    let m = msg.to_string();
    run_kernel("debug", move || -> scfast_core::Result<Robj> { panic!("{}", m) })
}

extendr_module! {
    mod scfastR;
    fn rs_pca_csc;
    fn rs_knn;
    fn rs_snn;
    fn rs_leiden;
    fn rs_is_symmetric;
    fn rs_info;
    fn rs_dataptr;
    fn rs_csr_view_ptrs;
    fn rs_debug_panic;
}
