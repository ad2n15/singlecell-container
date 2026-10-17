# scfastR: Seurat-facing wrappers around the scfast Rust kernels (docs/RUST_ENGINE.md).
#
#   PreprocessFast()    replaces NormalizeData() + FindVariableFeatures(vst) -> data layer, VariableFeatures
#   RunPCAFast()        replaces ScaleData() + RunPCA()     -> DimReduc "pca"
#   FindNeighborsFast() replaces FindNeighbors()            -> Graphs "<assay>_nn", "<assay>_snn"
#   FindClustersFast()  replaces FindClusters(algorithm=4)  -> seurat_clusters, Idents()
#
# Low-level, Seurat-free: scfast_pca(), scfast_knn(), scfast_snn(), scfast_leiden().
# The native calls (rs_*) are in R/extendr-wrappers.R; this file does argument checking,
# naming and the Seurat bookkeeping.

`%||%` <- function(a, b) if (is.null(a)) b else a

# ------------------------------------------------------------------ helpers

# Evaluate a native call; re-signal its error with classes "scfast_error" and, when a kernel
# is still a stub (unimplemented!()), "scfast_not_implemented".
.rust <- function(expr, call = sys.call(-1L)) {
  force(call)
  tryCatch(expr, error = function(e) {
    msg <- conditionMessage(e)
    cls <- c(if (grepl("not implemented", msg, fixed = TRUE)) "scfast_not_implemented",
             "scfast_error", "error", "condition")
    stop(structure(class = cls, list(message = msg, call = call)))
  })
}

.is_whole <- function(x) is.numeric(x) && all(is.finite(x)) && all(x == round(x))

.check_count <- function(x, name, min = 1) {
  if (!.is_whole(x) || length(x) != 1L || x < min) {
    stop(sprintf("'%s' must be a single whole number >= %s", name, min), call. = FALSE)
  }
  invisible(as.integer(x))
}

.check_seed <- function(x, name = "seed") {
  if (!.is_whole(x) || length(x) != 1L || x < 0) {
    stop(sprintf("'%s' must be a single non-negative whole number", name), call. = FALSE)
  }
  invisible(as.numeric(x))
}

.check_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("'%s' must be TRUE or FALSE", name), call. = FALSE)
  }
  invisible(x)
}

.check_string <- function(x, name) {
  if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(x)) {
    stop(sprintf("'%s' must be a single non-empty string", name), call. = FALSE)
  }
  invisible(x)
}

.check_seurat <- function(object) {
  if (!inherits(object, "Seurat")) {
    stop("'object' must be a Seurat object", call. = FALSE)
  }
  invisible(object)
}

# CUDA is a later phase (docs/RUST_ENGINE.md section 6); "auto" means CPU in this build.
.check_device <- function(device) {
  .check_string(device, "device")
  device <- tolower(device)
  if (!device %in% c("auto", "cpu", "cuda")) {
    stop("'device' must be \"auto\", \"cpu\" or \"cuda\"", call. = FALSE)
  }
  if (device == "cuda" && !isTRUE(rs_info()$cuda)) {
    stop("scfastR was built without CUDA support; use device = \"cpu\" or \"auto\"", call. = FALSE)
  }
  invisible("cpu")
}

# Any sparse/dense numeric matrix -> dgCMatrix. A dgCMatrix (or subclass, e.g. Graph) is
# returned as is: no copy.
.as_dgc <- function(m, name) {
  if (methods::is(m, "dgCMatrix")) return(m)
  if (!(is.matrix(m) || methods::is(m, "Matrix"))) {
    stop(sprintf("'%s' must be a dgCMatrix (or a matrix coercible to one)", name), call. = FALSE)
  }
  m <- methods::as(m, "CsparseMatrix")
  if (!methods::is(m, "generalMatrix")) m <- methods::as(m, "generalMatrix")
  if (!methods::is(m, "dMatrix")) m <- methods::as(m, "dMatrix")
  if (!methods::is(m, "dgCMatrix")) {
    stop(sprintf("could not convert '%s' to a dgCMatrix", name), call. = FALSE)
  }
  m
}

.timed <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  value <- expr
  list(value = value, seconds = proc.time()[["elapsed"]] - t0)
}

# ------------------------------------------------------------------ low-level API

#' Implicitly scaled PCA of a sparse expression matrix
#'
#' Runs the scfast randomised PCA on the centred and scaled matrix without building it:
#' the result matches `ScaleData()` + `RunPCA()` on the same features, except that
#' Seurat's `scale.max = 10` clipping is not applied.
#'
#' The `@p`, `@i` and `@x` arrays of `dgc` (genes x cells) are read in place by Rust as
#' the cells x genes CSR matrix they already are: no copy and no transpose.
#'
#' @param dgc A genes x cells `dgCMatrix` of log-normalised values (other matrix classes
#'   are converted, which copies).
#' @param npcs Number of components.
#' @param seed Random seed (non-negative whole number). Same seed, same result.
#' @param oversample,power.iters Randomised SVD settings; `NULL` keeps the engine defaults.
#' @return A list with `embeddings` (cells x npcs, `U * d` as in Seurat), `loadings`
#'   (genes x npcs), `stdev` (`d / sqrt(n_cells - 1)`), and the per-gene `gene_mean` and
#'   `gene_sd` used for scaling.
#' @export
scfast_pca <- function(dgc, npcs = 30, seed = 0, oversample = NULL, power.iters = NULL) {
  dgc <- .as_dgc(dgc, "dgc")
  npcs <- .check_count(npcs, "npcs")
  seed <- .check_seed(seed)
  if (!is.null(oversample)) .check_count(oversample, "oversample", min = 0)
  if (!is.null(power.iters)) .check_count(power.iters, "power.iters", min = 0)
  if (npcs >= min(dim(dgc))) {
    stop(sprintf("'npcs' (%d) must be less than min(genes, cells) = %d", npcs, min(dim(dgc))), call. = FALSE)
  }
  res <- .pca_native(dgc, npcs, seed, oversample, power.iters)
  pcs <- paste0("PC_", seq_len(ncol(res$embeddings)))
  dimnames(res$embeddings) <- list(colnames(dgc), pcs)
  dimnames(res$loadings) <- list(rownames(dgc), pcs)
  names(res$gene_mean) <- names(res$gene_sd) <- rownames(dgc)
  res
}

.pca_native <- function(dgc, npcs, seed, oversample = NULL, power.iters = NULL) {
  .rust(rs_pca_csc(dgc@p, dgc@i, dgc@x, nrow(dgc), ncol(dgc), as.integer(npcs), as.numeric(seed),
                   as.integer(oversample %||% -1L), as.integer(power.iters %||% -1L)))
}

#' k nearest neighbours of every row of a matrix
#'
#' @param mat Numeric matrix, cells x dimensions (e.g. PCA embeddings).
#' @param k Neighbours per cell, counting the cell itself (Seurat `k.param`); at least 2.
#' @param method `"auto"` (exact below the engine's cell-count threshold, HNSW above),
#'   `"exact"` or `"hnsw"`.
#' @param seed Seed for HNSW construction (exact search ignores it).
#' @param hnsw.m,ef.construction,ef.search HNSW settings, used with `method = "hnsw"`
#'   (`"auto"` uses the same defaults: 16, 200 and `max(2k, 64)`). HNSW results are not
#'   bit-reproducible across runs; exact search is.
#' @param auto.threshold Cell count where `"auto"` switches to HNSW; `NULL` keeps the
#'   engine default.
#' @return A list with `idx` (n x k integer, 1-based, the cell itself first) and `dist`
#'   (n x k Euclidean distances).
#' @export
scfast_knn <- function(mat, k = 20, method = "auto", seed = 0, hnsw.m = 16,
                       ef.construction = 200, ef.search = NULL, auto.threshold = NULL) {
  if (is.data.frame(mat)) mat <- as.matrix(mat)
  if (!is.matrix(mat) || !is.numeric(mat)) {
    stop("'mat' must be a numeric matrix (cells x dimensions)", call. = FALSE)
  }
  if (storage.mode(mat) != "double") storage.mode(mat) <- "double"
  k <- .check_count(k, "k", min = 2)
  .check_string(method, "method")
  if (!method %in% c("auto", "exact", "hnsw")) {
    stop("'method' must be one of \"auto\", \"exact\", \"hnsw\"", call. = FALSE)
  }
  seed <- .check_seed(seed)
  if (k > nrow(mat)) {
    stop(sprintf("'k' (%d) is larger than the number of cells (%d)", k, nrow(mat)), call. = FALSE)
  }
  hnsw.m <- .check_count(hnsw.m, "hnsw.m")
  ef.construction <- .check_count(ef.construction, "ef.construction")
  ef.search <- .check_count(ef.search %||% max(64L, 2L * k), "ef.search")
  if (!is.null(auto.threshold)) .check_count(auto.threshold, "auto.threshold", min = 0)
  res <- .rust(rs_knn(mat, nrow(mat), ncol(mat), k, method, seed, hnsw.m, ef.construction,
                      ef.search, as.numeric(auto.threshold %||% -1)))
  rownames(res$idx) <- rownames(res$dist) <- rownames(mat)
  res
}

#' Shared-nearest-neighbour graph from kNN indices
#'
#' Seurat's `ComputeSNN()`: Jaccard index of the neighbour sets (each including the cell
#' itself), entries below `prune` removed, diagonal kept.
#'
#' @param idx Integer matrix n x k of 1-based neighbour indices, the cell itself first (the
#'   `idx` element of [scfast_knn()]).
#' @param prune Seurat `prune.SNN`.
#' @return A symmetric n x n `dgCMatrix`, named by `rownames(idx)`.
#' @export
scfast_snn <- function(idx, prune = 1 / 15) {
  if (!is.matrix(idx) || !.is_whole(idx)) {
    stop("'idx' must be a matrix of whole numbers (1-based neighbour indices)", call. = FALSE)
  }
  if (storage.mode(idx) != "integer") storage.mode(idx) <- "integer"
  if (!is.numeric(prune) || length(prune) != 1L || !is.finite(prune) || prune < 0 || prune > 1) {
    stop("'prune' must be a single number in [0, 1]", call. = FALSE)
  }
  g <- .snn_native(idx, prune)
  if (!is.null(rownames(idx))) g@Dimnames <- list(rownames(idx), rownames(idx))
  g
}

.snn_native <- function(idx, prune) {
  parts <- .rust(rs_snn(idx, nrow(idx), ncol(idx), as.numeric(prune)))
  methods::new("dgCMatrix", i = parts$i, p = parts$p, x = parts$x, Dim = c(parts$n, parts$n))
}

#' Leiden clustering of a graph
#'
#' Leiden with the RB objective (what Seurat's `algorithm = 4` and Scanpy optimise): on the
#' CPU with single-clustering, or on a GPU with cuGraph when scfastR was built with the
#' `cugraph` feature and `libcugraph_c.so` loads. A non-symmetric graph is symmetrised as
#' `(G + t(G)) / 2`, with a warning.
#'
#' @param graph Square weighted adjacency (`dgCMatrix` or Seurat `Graph`), e.g.
#'   `object[["RNA_snn"]]`.
#' @param resolution RB resolution (Seurat `resolution`).
#' @param seed Random seed (Seurat `random.seed`).
#' @param n.iter Leiden iterations (Seurat `n.iter`, leidenalg `n_iterations`; cuGraph
#'   `max_level`).
#' @param device `"auto"` (cuGraph when available, else CPU), `"cpu"`, `"cuda"` or
#'   `"cuda:<index>"` (an error when cuGraph is not available). The device used is returned
#'   in the `"device"` attribute.
#' @return Integer vector of 0-based labels numbered by decreasing cluster size, named by
#'   `colnames(graph)`, with attribute `device` (`"cpu"` or `"cuda:<index>"`).
#' @export
scfast_leiden <- function(graph, resolution = 0.5, seed = 0, n.iter = 2, device = "auto") {
  device <- .check_leiden_device(device)
  graph <- .prepare_graph(graph)
  .check_resolution(resolution)
  if (length(resolution) != 1L) stop("'resolution' must be a single number", call. = FALSE)
  labels <- .leiden_native(graph, resolution, .check_seed(seed), .check_count(n.iter, "n.iter"), device)
  names(labels) <- colnames(graph)
  attr(labels, "device") <- device
  labels
}

# Leiden device string -> the device the native side will run on ("cpu" or "cuda:<i>").
# "auto" resolves to cuGraph only when scfastR was built with the `cugraph` feature, the
# library loads and a CUDA device is usable; an explicit "cuda" that is not usable errors.
# Separate from .check_device() (PCA / kNN, Phase 6): the two GPU paths are built and
# detected independently.
.check_leiden_device <- function(device) {
  .check_string(device, "device")
  device <- tolower(device)
  if (!grepl("^(auto|cpu|cuda|gpu)(:[0-9]+)?$", device) || device %in% c("auto:0", "cpu:0")) {
    stop("'device' must be \"auto\", \"cpu\", \"cuda\" or \"cuda:<index>\"", call. = FALSE)
  }
  .rust(rs_leiden_device(device))$resolved
}

.check_resolution <- function(resolution) {
  if (!is.numeric(resolution) || length(resolution) < 1L || any(!is.finite(resolution)) || any(resolution < 0)) {
    stop("'resolution' must be non-negative number(s)", call. = FALSE)
  }
  invisible(resolution)
}

.prepare_graph <- function(graph) {
  graph <- .as_dgc(graph, "graph")
  if (nrow(graph) != ncol(graph)) stop("'graph' must be a square matrix", call. = FALSE)
  if (!.rust(rs_is_symmetric(graph@p, graph@i, graph@x, ncol(graph), 1e-6))) {
    warning("graph is not symmetric; clustering (G + t(G)) / 2", call. = FALSE)
    graph <- .as_dgc((graph + Matrix::t(graph)) / 2, "graph")
  }
  graph
}

.leiden_native <- function(graph, resolution, seed, n.iter, device = "cpu") {
  .rust(rs_leiden_on(graph@p, graph@i, graph@x, ncol(graph), as.numeric(resolution),
                     as.numeric(seed), as.integer(n.iter), device))
}

#' Engine information
#'
#' @return A list: package and engine versions, rayon thread count (set the environment
#'   variable `RAYON_NUM_THREADS` before the first scfast call to change it), and whether
#'   CUDA was compiled in.
#' @export
scfast_info <- function() {
  c(list(package = as.character(utils::packageVersion("scfastR"))), rs_info())
}

# ------------------------------------------------------------------ Seurat: PCA

# The assay's "data" layer exactly as stored (no LayerData() subsetting, which copies),
# with its feature and cell names.
.data_layer <- function(object, assay) {
  a <- object[[assay]]
  if (inherits(a, "Assay5")) {
    layers <- SeuratObject::Layers(a)
    if (!"data" %in% layers) {
      if (any(startsWith(layers, "data."))) {
        stop(sprintf("assay '%s' has split data layers (%s); run JoinLayers() first",
                     assay, paste(grep("^data\\.", layers, value = TRUE), collapse = ", ")), call. = FALSE)
      }
      stop(sprintf("assay '%s' has no 'data' layer; run NormalizeData() first", assay), call. = FALSE)
    }
    m <- methods::slot(a, "layers")[["data"]]
    features <- SeuratObject::Features(a, layer = "data")
    cells <- SeuratObject::Cells(a, layer = "data")
  } else if (inherits(a, "Assay")) {
    m <- methods::slot(a, "data")
    features <- rownames(m)
    cells <- colnames(m)
  } else {
    stop(sprintf("assay '%s' has unsupported class '%s'", assay, class(a)[1]), call. = FALSE)
  }
  if (!identical(dim(m), c(length(features), length(cells)))) {
    stop(sprintf("assay '%s': the data layer is not stored as features x cells", assay), call. = FALSE)
  }
  list(m = m, features = features, cells = cells)
}

# The genes x cells dgCMatrix passed to Rust: the data layer restricted to `features`
# (one copy of the HVG rows; none when all features are used in stored order).
.pca_input <- function(object, assay, features = NULL, verbose = FALSE) {
  layer <- .data_layer(object, assay)
  if (is.null(features)) {
    features <- SeuratObject::VariableFeatures(object, assay = assay)
    if (length(features) == 0L) {
      stop(sprintf("assay '%s' has no variable features; run FindVariableFeatures() first or pass 'features'", assay),
           call. = FALSE)
    }
  }
  if (!is.character(features)) stop("'features' must be a character vector", call. = FALSE)
  features <- unique(features)
  idx <- match(features, layer$features)
  if (anyNA(idx)) {
    warning(sprintf("%d of %d requested features are not in assay '%s' and are skipped: %s",
                    sum(is.na(idx)), length(features), assay,
                    paste(utils::head(features[is.na(idx)], 10), collapse = ", ")), call. = FALSE)
    features <- features[!is.na(idx)]
    idx <- idx[!is.na(idx)]
  }
  if (length(idx) < 2L) stop("fewer than 2 usable features", call. = FALSE)
  m <- layer$m
  if (!identical(idx, seq_len(nrow(m)))) m <- m[idx, , drop = FALSE]
  m <- .as_dgc(m, "data layer")
  # Seurat drops zero-variance features before PCA; all-zero rows are the case that occurs.
  nnz <- tabulate(m@i + 1L, nbins = nrow(m))
  if (any(nnz == 0L)) {
    keep <- nnz > 0L
    warning(sprintf("%d features have zero variance and are skipped: %s", sum(!keep),
                    paste(utils::head(features[!keep], 10), collapse = ", ")), call. = FALSE)
    m <- m[keep, , drop = FALSE]
    features <- features[keep]
  }
  list(m = m, features = features, cells = layer$cells)
}

#' PCA on the log-normalised data with implicit scaling (no ScaleData)
#'
#' Replaces `ScaleData()` + `RunPCA()`. The assay's `data` layer restricted to the variable
#' features is passed to Rust in place, and the scaled matrix is never built. Stores a
#' standard `DimReduc`, so `ElbowPlot()`, `DimPlot()`, `FindNeighbors()`, `RunUMAP()`
#' work unchanged.
#'
#' @param object A Seurat object (after `NormalizeData()` and `FindVariableFeatures()`).
#' @param assay Assay to use; default `DefaultAssay(object)`.
#' @param features Features to use; default `VariableFeatures()`.
#' @param npcs Number of components (capped at `min(features, cells) - 1`, as in Seurat).
#' @param seed.use Random seed (`seed` also works, by partial matching).
#' @param reduction.name,reduction.key Name and key of the stored reduction.
#' @param device `"auto"`, `"cpu"` or `"cuda"` (CUDA is not built yet; `"auto"` = CPU).
#' @param oversample,power.iters Randomised SVD accuracy knobs (extra random directions and
#'   power iterations); `NULL` keeps the engine defaults (10 and 4). Raise them (e.g. 40 and
#'   8) when the spectrum is flat beyond the leading components and later PCs must match
#'   an exact SVD closely; cost grows roughly linearly with `power.iters`.
#' @param verbose Print progress messages.
#' @return The Seurat object with the reduction added.
#' @export
RunPCAFast <- function(object, assay = NULL, features = NULL, npcs = 30, seed.use = 0,
                       reduction.name = "pca", reduction.key = "PC_", device = "auto",
                       oversample = NULL, power.iters = NULL, verbose = TRUE) {
  .check_seurat(object)
  assay <- assay %||% SeuratObject::DefaultAssay(object)
  .check_string(assay, "assay")
  if (!assay %in% SeuratObject::Assays(object)) {
    stop(sprintf("assay '%s' not found in the object", assay), call. = FALSE)
  }
  npcs <- .check_count(npcs, "npcs")
  seed.use <- .check_seed(seed.use, "seed.use")
  .check_string(reduction.name, "reduction.name")
  .check_string(reduction.key, "reduction.key")
  .check_device(device)
  if (!is.null(oversample)) .check_count(oversample, "oversample", min = 0)
  if (!is.null(power.iters)) .check_count(power.iters, "power.iters", min = 0)
  .check_flag(verbose, "verbose")

  input <- .pca_input(object, assay, features, verbose)
  max.pcs <- min(dim(input$m)) - 1L
  if (max.pcs < 1L) stop("need at least 2 features and 2 cells", call. = FALSE)
  if (npcs > max.pcs) {
    warning(sprintf("npcs (%d) reduced to %d (min(features, cells) - 1)", npcs, max.pcs), call. = FALSE)
    npcs <- max.pcs
  }
  run <- .timed(.pca_native(input$m, npcs, seed.use, oversample, power.iters))
  res <- run$value
  pcs <- paste0(reduction.key, seq_len(ncol(res$embeddings)))
  dimnames(res$embeddings) <- list(input$cells, pcs)
  dimnames(res$loadings) <- list(input$features, pcs)
  names(res$gene_mean) <- names(res$gene_sd) <- input$features
  reduc <- SeuratObject::CreateDimReducObject(
    embeddings = res$embeddings,
    loadings = res$loadings,
    stdev = res$stdev,
    key = reduction.key,
    assay = assay,
    misc = list(
      # Every scaled feature has unit variance (no clipping), as total.variance in RunPCA.
      total.variance = length(input$features),
      scfast = list(kernel = "implicit_scaled_pca", engine = scfast_info()$version,
                    seed = seed.use, oversample = oversample, power.iters = power.iters,
                    seconds = run$seconds,
                    gene_mean = res$gene_mean, gene_sd = res$gene_sd)
    )
  )
  if (verbose) {
    message(sprintf("scfast PCA: %d cells x %d features, %d PCs in %.2f s (no ScaleData)",
                    length(input$cells), length(input$features), npcs, run$seconds))
  }
  object[[reduction.name]] <- reduc
  object <- SeuratObject::LogSeuratCommand(object = object)
  object
}

# ------------------------------------------------------------------ Seurat: neighbours

#' kNN and SNN graphs on a reduction
#'
#' Replaces `FindNeighbors()` (Annoy + `ComputeSNN`). Stores the kNN graph (0/1 adjacency,
#' the cell itself included) and the SNN graph (Jaccard weights, pruned) as Seurat `Graph`
#' objects named `<assay>_nn` and `<assay>_snn`, exactly where `FindNeighbors()` puts them.
#'
#' @param object A Seurat object with the reduction.
#' @param reduction Reduction to use.
#' @param dims Dimensions of the reduction to use.
#' @param k.param Neighbours per cell, including the cell itself.
#' @param prune.SNN SNN weights below this are set to zero.
#' @param method `"auto"`, `"exact"` or `"hnsw"` (see [scfast_knn()]).
#' @param graph.name `NULL` (default names), or two names (kNN graph, SNN graph). As in
#'   Seurat, a single name stores only the kNN graph.
#' @param random.seed Seed for HNSW (exact search ignores it).
#' @param device `"auto"`, `"cpu"` or `"cuda"` (CUDA is not built yet).
#' @param verbose Print progress messages.
#' @return The Seurat object with the graphs added.
#' @export
FindNeighborsFast <- function(object, reduction = "pca", dims = 1:10, k.param = 20,
                              prune.SNN = 1 / 15, method = "auto", graph.name = NULL,
                              random.seed = 0, device = "auto", verbose = TRUE) {
  .check_seurat(object)
  .check_string(reduction, "reduction")
  if (!reduction %in% SeuratObject::Reductions(object)) {
    stop(sprintf("reduction '%s' not found; run RunPCAFast() or RunPCA() first", reduction), call. = FALSE)
  }
  k.param <- .check_count(k.param, "k.param", min = 2)
  if (!is.numeric(prune.SNN) || length(prune.SNN) != 1L || !is.finite(prune.SNN) || prune.SNN < 0 || prune.SNN > 1) {
    stop("'prune.SNN' must be a single number in [0, 1]", call. = FALSE)
  }
  .check_string(method, "method")
  if (!method %in% c("auto", "exact", "hnsw")) {
    stop("'method' must be one of \"auto\", \"exact\", \"hnsw\"", call. = FALSE)
  }
  random.seed <- .check_seed(random.seed, "random.seed")
  .check_device(device)
  .check_flag(verbose, "verbose")

  reduc <- object[[reduction]]
  assay <- SeuratObject::DefaultAssay(reduc)
  emb <- SeuratObject::Embeddings(reduc)
  if (!.is_whole(dims) || length(dims) < 1L || any(dims < 1)) {
    stop("'dims' must be positive whole numbers", call. = FALSE)
  }
  if (max(dims) > ncol(emb)) stop("More dimensions specified in dims than have been computed", call. = FALSE)
  if (anyDuplicated(dims)) stop("'dims' contains duplicates", call. = FALSE)
  data.use <- emb[, dims, drop = FALSE]
  cells <- rownames(data.use)
  n <- nrow(data.use)
  if (n < k.param) {
    warning("k.param set larger than number of cells. Setting k.param to number of cells - 1.", call. = FALSE)
    k.param <- n - 1L
    if (k.param < 2L) stop("need at least 3 cells", call. = FALSE)
  }
  if (is.null(graph.name)) {
    graph.name <- paste0(assay, "_", c("nn", "snn"))
  } else if (!is.character(graph.name) || !length(graph.name) %in% 1:2 || anyNA(graph.name)) {
    stop("'graph.name' must be NULL or one or two names", call. = FALSE)
  }
  if (length(graph.name) == 1L) {
    message("Only one graph name supplied, storing nearest-neighbor graph only")
  }

  t0 <- proc.time()[["elapsed"]]
  nn <- scfast_knn(data.use, k = k.param, method = method, seed = random.seed)
  t.knn <- proc.time()[["elapsed"]] - t0
  # Same construction as Seurat: row i has a 1 at each of its k neighbours (itself included).
  nn.graph <- Matrix::sparseMatrix(i = rep(seq_len(n), times = k.param), j = as.vector(nn$idx),
                                   x = 1, dims = c(n, n), dimnames = list(cells, cells))
  graphs <- list(SeuratObject::as.Graph(nn.graph))
  t.snn <- 0
  if (length(graph.name) == 2L) {
    t1 <- proc.time()[["elapsed"]]
    snn <- .snn_native(nn$idx, prune.SNN)
    snn@Dimnames <- list(cells, cells)
    graphs[[2L]] <- SeuratObject::as.Graph(snn)
    t.snn <- proc.time()[["elapsed"]] - t1
  }
  for (ii in seq_along(graph.name)) {
    SeuratObject::DefaultAssay(graphs[[ii]]) <- assay
    object[[graph.name[[ii]]]] <- graphs[[ii]]
  }
  if (verbose) {
    message(sprintf("scfast neighbours: %d cells, k = %d, %d dims: kNN %.2f s, SNN %.2f s",
                    n, k.param, length(dims), t.knn, t.snn))
  }
  object <- SeuratObject::LogSeuratCommand(object = object)
  object
}

# ------------------------------------------------------------------ Seurat: clustering

# Seurat's GroupSingletons(): each cluster of one cell joins the cluster it is most connected
# to (mean SNN weight to that cluster's cells), processed in order of first appearance; ties
# are broken as Seurat does (set.seed(1); sample()), but the caller's RNG state is restored.
.group_singletons <- function(labels, graph, verbose) {
  sizes <- tabulate(labels + 1L, nbins = max(labels) + 1L)
  singletons <- intersect(unique(labels), which(sizes == 1L) - 1L)
  if (length(singletons) == 0L) return(labels)
  cluster.ids <- setdiff(unique(labels), singletons)
  if (length(cluster.ids) == 0L) return(labels)
  had.seed <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  if (had.seed) old.seed <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
  on.exit(if (had.seed) assign(".Random.seed", old.seed, envir = globalenv())
          else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE))
            rm(".Random.seed", envir = globalenv()), add = TRUE)
  for (s in singletons) {
    cell <- which(labels == s)
    rng <- if (graph@p[cell] < graph@p[cell + 1L]) seq.int(graph@p[cell] + 1L, graph@p[cell + 1L]) else integer()
    nbr <- graph@i[rng] + 1L
    w <- graph@x[rng]
    keep <- nbr != cell
    nbr.lab <- labels[nbr[keep]]
    w <- w[keep]
    sizes <- tabulate(labels + 1L, nbins = max(labels) + 1L)
    conn <- vapply(cluster.ids, function(j) sum(w[nbr.lab == j]) / sizes[j + 1L], numeric(1))
    best <- cluster.ids[which(conn == max(conn))]
    if (length(best) > 1L) {
      set.seed(1)
      best <- best[sample.int(length(best), 1L)]
    }
    labels[cell] <- best
  }
  if (verbose) {
    message(paste(length(singletons), "singletons identified.", length(unique(labels)), "final clusters."))
  }
  labels
}

#' Leiden clustering of the SNN graph
#'
#' Replaces `FindClusters(algorithm = 4)` (leidenalg, RB objective) with the scfast Leiden
#' (CPU, or cuGraph on a GPU with `device`). Writes the same outputs: a meta data column
#' `<graph.name>_res.<resolution>` per resolution, `seurat_clusters`, and `Idents()` (factor
#' with levels 0..n-1).
#'
#' @param object A Seurat object with an SNN graph (from `FindNeighborsFast()` or
#'   `FindNeighbors()`).
#' @param graph.name Graph to cluster; default `<DefaultAssay>_snn`.
#' @param resolution Resolution(s); one column per value, `Idents()` from the last.
#' @param random.seed Random seed (`seed = ` is accepted as an alias).
#' @param n.iter Leiden iterations.
#' @param group.singletons Merge single-cell clusters into their most connected cluster,
#'   as Seurat does.
#' @param device `"auto"` (cuGraph Leiden when scfastR was built with the `cugraph` feature
#'   and a CUDA device is usable, else CPU), `"cpu"`, `"cuda"` or `"cuda:<index>"`.
#' @param verbose Print progress messages.
#' @param ... `seed` (alias of `random.seed`); `algorithm` (only 4 / "leiden") and `method`
#'   (ignored) are accepted so a `FindClusters()` call can be renamed unchanged.
#' @return The Seurat object with clusters added.
#' @export
FindClustersFast <- function(object, graph.name = NULL, resolution = 0.5, random.seed = 0,
                             n.iter = 2, group.singletons = TRUE, device = "auto", verbose = TRUE, ...) {
  .check_seurat(object)
  device <- .check_leiden_device(device)
  dots <- list(...)
  if (length(dots) && (is.null(names(dots)) || any(!nzchar(names(dots))))) {
    stop("unnamed arguments in '...'", call. = FALSE)
  }
  unknown <- setdiff(names(dots), c("seed", "algorithm", "method"))
  if (length(unknown)) {
    stop(sprintf("unused argument(s): %s", paste(unknown, collapse = ", ")), call. = FALSE)
  }
  if (!is.null(dots$seed)) {
    if (!missing(random.seed)) stop("give either 'random.seed' or 'seed', not both", call. = FALSE)
    random.seed <- dots$seed
  }
  if (!is.null(dots$algorithm) && !(tolower(as.character(dots$algorithm)) %in% c("4", "leiden"))) {
    stop("FindClustersFast() implements Leiden only (algorithm = 4)", call. = FALSE)
  }
  graph.name <- graph.name %||% paste0(SeuratObject::DefaultAssay(object), "_snn")
  .check_string(graph.name, "graph.name")
  .check_resolution(resolution)
  random.seed <- .check_seed(random.seed, "random.seed")
  n.iter <- .check_count(n.iter, "n.iter")
  .check_flag(group.singletons, "group.singletons")
  .check_flag(verbose, "verbose")
  if (!graph.name %in% names(object)) {
    stop("Provided graph.name not present in Seurat object", call. = FALSE)
  }
  graph <- object[[graph.name]]
  if (!methods::is(graph, "Graph")) {
    stop("Provided graph.name does not correspond to a graph object.", call. = FALSE)
  }
  cells <- colnames(graph)
  if (!setequal(cells, SeuratObject::Cells(object))) {
    stop(sprintf("graph '%s' does not cover the same cells as the object", graph.name), call. = FALSE)
  }
  graph <- .prepare_graph(graph)

  results <- data.frame(row.names = cells)
  for (r in resolution) {
    run <- .timed(.leiden_native(graph, r, random.seed, n.iter, device))
    labels <- run$value
    if (group.singletons) labels <- .group_singletons(labels, graph, verbose)
    col <- paste0(graph.name, "_res.", r)
    results[[col]] <- factor(labels, levels = sort(unique(labels)))
    if (verbose) {
      message(sprintf("scfast Leiden (%s): %d clusters at resolution %s in %.2f s",
                      device, length(unique(labels)), format(r), run$seconds))
    }
  }
  object <- SeuratObject::AddMetaData(object, metadata = results)
  last <- names(results)[ncol(results)]
  SeuratObject::Idents(object = object, replace = TRUE) <- object[[last, drop = TRUE]]
  lv <- levels(SeuratObject::Idents(object))
  SeuratObject::Idents(object) <- factor(SeuratObject::Idents(object), levels = lv[order(as.numeric(lv))])
  object[["seurat_clusters"]] <- SeuratObject::Idents(object)
  object <- SeuratObject::LogSeuratCommand(object = object)
  object
}

# ------------------------------------------------------------------ preprocessing (Phase 9)

.check_positive_number <- function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x <= 0) {
    stop(sprintf("'%s' must be a single positive number", name), call. = FALSE)
  }
  invisible(as.numeric(x))
}

#' Fused log-normalisation and vst variable-feature statistics of a counts matrix
#'
#' One streaming pass over the raw counts computes what `NormalizeData()`,
#' `FindVariableFeatures(selection.method = "vst")` and the per-gene statistics of
#' `ScaleData()` compute: per-cell library sizes, `log1p(counts / library_size *
#' scale.factor)`, the vst mean / variance / loess-expected variance / clipped standardised
#' variance of every gene, the top `n_hvg` genes by standardised variance, and the mean and
#' standard deviation of their log-normalised values (what the implicit PCA scales with).
#'
#' The `@p`, `@i` and `@x` arrays of `counts` (genes x cells) are read in place; the
#' returned `data` matrix shares `@p` and `@i` with `counts`.
#'
#' The loess (local quadratic, tricube, span `loess.span`) is R's
#' `loess(log10(variance) ~ log10(mean), surface = "direct")`; Seurat uses the default
#' interpolated surface, so expected variances differ slightly (HVG overlap > 99% on the
#' test data).
#'
#' @param counts A genes x cells `dgCMatrix` of raw counts (other matrix classes are
#'   converted, which copies).
#' @param n_hvg Number of variable features (Seurat `nfeatures`).
#' @param scale.factor `NormalizeData(scale.factor)`.
#' @param loess.span `FindVariableFeatures(loess.span)`.
#' @param clip.max Clip for the standardised values; `NULL` (default) means
#'   `sqrt(n_cells)`, as Seurat's `clip.max = "auto"`.
#' @return A list: `data` (genes x cells `dgCMatrix` of log-normalised values),
#'   `library_size` (per cell, named), `hvg` (feature names, most variable first),
#'   `hvg_stats` (data frame with one row per gene: `mean`, `variance`, `variance.expected`,
#'   `variance.standardized`, `variable`, `rank`), `hvg_mean` and `hvg_sd` (named by HVG),
#'   `clip` and `seconds`.
#' @export
scfast_preprocess <- function(counts, n_hvg = 2000, scale.factor = 1e4, loess.span = 0.3, clip.max = NULL) {
  counts <- .as_dgc(counts, "counts")
  n_hvg <- .check_count(n_hvg, "n_hvg")
  scale.factor <- .check_positive_number(scale.factor, "scale.factor")
  loess.span <- .check_positive_number(loess.span, "loess.span")
  clip <- if (is.null(clip.max)) -1 else .check_positive_number(clip.max, "clip.max")
  if (nrow(counts) < 1L || ncol(counts) < 1L) stop("'counts' is empty", call. = FALSE)
  if (n_hvg > nrow(counts)) {
    warning(sprintf("n_hvg (%d) reduced to the number of features (%d)", n_hvg, nrow(counts)), call. = FALSE)
    n_hvg <- nrow(counts)
  }
  run <- .timed(.preprocess_native(counts, n_hvg, scale.factor, loess.span, clip))
  .preprocess_result(run$value, counts, run$seconds)
}

.preprocess_native <- function(counts, n_hvg, scale.factor, loess.span, clip) {
  .rust(rs_preprocess_csc(counts@p, counts@i, counts@x, nrow(counts), ncol(counts), as.integer(n_hvg),
                          as.numeric(scale.factor), as.numeric(loess.span), as.numeric(clip)))
}

# Native result -> the documented list (data dgCMatrix sharing @p / @i with the counts).
.preprocess_result <- function(res, counts, seconds) {
  features <- rownames(counts) %||% paste0("feature", seq_len(nrow(counts)))
  cells <- colnames(counts)
  data <- methods::new("dgCMatrix", i = counts@i, p = counts@p, x = res$data, Dim = dim(counts),
                       Dimnames = list(rownames(counts), cells))
  rank <- rep(NA_integer_, nrow(counts))
  rank[res$hvg] <- seq_along(res$hvg)
  stats <- data.frame(mean = res$mean, variance = res$variance, variance.expected = res$variance_expected,
                      variance.standardized = res$variance_standardized, variable = !is.na(rank), rank = rank,
                      row.names = features, stringsAsFactors = FALSE)
  lib <- res$library_size
  names(lib) <- cells
  hvg <- features[res$hvg]
  hvg_mean <- res$hvg_mean
  hvg_sd <- res$hvg_sd
  names(hvg_mean) <- names(hvg_sd) <- hvg
  list(data = data, library_size = lib, hvg = hvg, hvg_stats = stats, hvg_mean = hvg_mean, hvg_sd = hvg_sd,
       clip = res$clip, seconds = seconds)
}

# The assay's "counts" layer exactly as stored, with feature and cell names.
.counts_layer <- function(object, assay) {
  a <- object[[assay]]
  if (inherits(a, "Assay5")) {
    layers <- SeuratObject::Layers(a)
    if (!"counts" %in% layers) {
      if (any(startsWith(layers, "counts."))) {
        stop(sprintf("assay '%s' has split counts layers (%s); run JoinLayers() first",
                     assay, paste(grep("^counts\\.", layers, value = TRUE), collapse = ", ")), call. = FALSE)
      }
      stop(sprintf("assay '%s' has no 'counts' layer", assay), call. = FALSE)
    }
    m <- methods::slot(a, "layers")[["counts"]]
    features <- SeuratObject::Features(a, layer = "counts")
    cells <- SeuratObject::Cells(a, layer = "counts")
  } else if (inherits(a, "Assay")) {
    m <- methods::slot(a, "counts")
    features <- rownames(m)
    cells <- colnames(m)
  } else {
    stop(sprintf("assay '%s' has unsupported class '%s'", assay, class(a)[1]), call. = FALSE)
  }
  if (!identical(dim(m), c(length(features), length(cells)))) {
    stop(sprintf("assay '%s': the counts layer is not stored as features x cells", assay), call. = FALSE)
  }
  if (length(cells) != ncol(object) || !setequal(cells, SeuratObject::Cells(object))) {
    stop(sprintf("assay '%s': the counts layer does not cover all cells of the object", assay), call. = FALSE)
  }
  m <- .as_dgc(m, "counts layer")
  if (is.null(dimnames(m)) || is.null(rownames(m))) dimnames(m) <- list(features, cells)
  list(m = m, features = features, cells = cells)
}

#' Normalise, find variable features and compute scaling statistics in one pass
#'
#' Replaces `NormalizeData(normalization.method = "LogNormalize")` +
#' `FindVariableFeatures(selection.method = "vst")` (+ the per-gene statistics of
#' `ScaleData()`, which `RunPCAFast()` applies implicitly) with one streaming pass over the
#' assay's counts (see [scfast_preprocess()]). Writes the `data` layer, the variable
#' features (and the `vf_vst_counts_*` feature meta data that `HVFInfo()` and
#' `VariableFeaturePlot()` read), and `Misc(object, "scfast")$hvg_stats`.
#'
#' @param object A Seurat object with raw counts.
#' @param n_hvg Number of variable features (Seurat `nfeatures`).
#' @param assay Assay to use; default `DefaultAssay(object)`.
#' @param scale.factor `NormalizeData(scale.factor)`.
#' @param loess.span `FindVariableFeatures(loess.span)`.
#' @param clip.max `NULL` (`sqrt(n_cells)`, Seurat's `"auto"`) or a positive number.
#' @param verbose Print a progress message.
#' @return The Seurat object with the `data` layer, variable features and statistics added.
#' @export
PreprocessFast <- function(object, n_hvg = 2000, assay = NULL, scale.factor = 1e4, loess.span = 0.3,
                           clip.max = NULL, verbose = TRUE) {
  .check_seurat(object)
  assay <- assay %||% SeuratObject::DefaultAssay(object)
  .check_string(assay, "assay")
  if (!assay %in% SeuratObject::Assays(object)) {
    stop(sprintf("assay '%s' not found in the object", assay), call. = FALSE)
  }
  n_hvg <- .check_count(n_hvg, "n_hvg")
  scale.factor <- .check_positive_number(scale.factor, "scale.factor")
  loess.span <- .check_positive_number(loess.span, "loess.span")
  clip <- if (is.null(clip.max)) -1 else .check_positive_number(clip.max, "clip.max")
  .check_flag(verbose, "verbose")

  layer <- .counts_layer(object, assay)
  n_feat <- nrow(layer$m)
  if (n_hvg > n_feat) {
    warning(sprintf("n_hvg (%d) reduced to the number of features (%d)", n_hvg, n_feat), call. = FALSE)
    n_hvg <- n_feat
  }
  run <- .timed(.preprocess_native(layer$m, n_hvg, scale.factor, loess.span, clip))
  res <- .preprocess_result(run$value, layer$m, run$seconds)

  a <- object[[assay]]
  v5 <- inherits(a, "Assay5")
  SeuratObject::LayerData(a, layer = "data") <- res$data
  # Feature meta data in Seurat's own column names, so HVFInfo() / VariableFeaturePlot() work.
  stats <- res$hvg_stats
  prefix <- if (v5) "vf_vst_counts_" else "vst."
  cols <- paste0(prefix, c("mean", "variance", "variance.expected", "variance.standardized", "variable", "rank"))
  if (v5) {
    a[["var.features"]] <- NULL
    a[["var.features.rank"]] <- NULL
  }
  for (k in seq_along(cols)) {
    vals <- stats[[k]]
    names(vals) <- rownames(stats)
    a[[cols[k]]] <- vals
  }
  SeuratObject::VariableFeatures(a) <- res$hvg
  object[[assay]] <- a
  misc <- SeuratObject::Misc(object, slot = "scfast") %||% list()
  misc$hvg_stats <- stats
  misc$preprocess <- list(kernel = "preprocess", engine = scfast_info()$version, assay = assay,
                          n_hvg = n_hvg, scale.factor = scale.factor, loess.span = loess.span, clip = res$clip,
                          seconds = run$seconds, library_size = res$library_size,
                          hvg_mean = res$hvg_mean, hvg_sd = res$hvg_sd)
  SeuratObject::Misc(object, slot = "scfast") <- misc
  if (verbose) {
    message(sprintf("scfast preprocess: %d cells x %d features, %d variable features in %.2f s (one pass: normalise + vst + scaling stats)",
                    length(layer$cells), n_feat, length(res$hvg), run$seconds))
  }
  object <- SeuratObject::LogSeuratCommand(object = object)
  object
}
