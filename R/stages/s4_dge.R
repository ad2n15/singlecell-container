# scpipe stage S4: pseudobulk differential expression (docs/agents-spec.md
# section 6, DESIGN.md 6.1). The Open Targets half of S4 is Python
# (python/scpipe/stages/s4_opentargets.py) and runs after this step.
#
# Inputs:  S2/object.rds (or .qs2); optional gates/G3/decision.json or
#          S3/annotation.tsv (cell-type labels)
# Outputs: S3/annotation.tsv                      (written from gates/G3/decision.json when it exists)
#          S4/gene_map.tsv                        gene, symbol, ensembl_id for every feature
#          S4/pseudobulk/<contrast>/counts.tsv.gz gene x pseudobulk unit (group x sample) counts
#          S4/pseudobulk/<contrast>/samples.tsv   one row per unit: pb_id, group, sample, condition,
#                                                 covariates, n_cells, included, exclude_reason
#          S4/dge/<contrast>__<group>.tsv.gz      gene, baseMean, log2fc, lfcSE, pvalue, padj, direction
#          S4/dge_summary.json                    per contrast x group: n_up, n_down, n_tested,
#                                                 skipped_reason, top 10 up/down, and the choices made
#
# Choices not fixed by the spec (documented here):
# * Labels: if gates/G3/decision.json exists its clusters are written to
#   S3/annotation.tsv and added to the object as cell_type_broad / cell_type_fine.
#   Without it an existing S3/annotation.tsv is used; without either, group_by
#   falls back to seurat_clusters with a note in dge_summary.json.
#   `group_by: cluster` means seurat_clusters.
# * Pseudobulk unit = group_by x sample_col. The two columns are combined into
#   one temporary metadata column before Seurat::AggregateExpression so that
#   Seurat's renaming of group names (underscores to dashes, a "g" prefix for
#   names starting with a digit) can never merge or mislabel units; the
#   grouping is still group_by x sample_col. If AggregateExpression refuses
#   (e.g. a single unit) the sums are computed directly, which gives the same
#   matrix.
# * Units with fewer than min_cells_per_sample cells are dropped before
#   counting replicates. A group needs min_replicates units on each side of
#   the contrast, otherwise it is skipped with skipped_reason.
# * Design: ~ covariates + condition, condition levels (denominator,
#   numerator). A covariate that has one level within a group, or that makes
#   the design rank deficient or leaves no residual degrees of freedom, is
#   dropped for that group (recorded in covariates_used / notes).
# * Genes with fewer than 10 summed counts across the group's units are not
#   tested (n_tested counts the rest); they are absent from the DGE table.
# * method deseq2: DESeq2 with Wald test, results(contrast = c(condition,
#   numerator, denominator)). method limma_voom (also the fallback when
#   DESeq2 is not installed, or when DESeq2 fails on a group): limma::voom on
#   library-size-normalised counts (no edgeR TMM factors; edgeR is not
#   required), lmFit + eBayes; baseMean is the mean library-size-normalised
#   count, lfcSE = sqrt(s2.post) * stdev.unscaled.
# * direction: up if padj < dge.padj and log2fc >= dge.lfc; down if padj < dge.padj
#   and log2fc <= -dge.lfc; otherwise ns (NA padj counts as ns).

.S4_DEFAULTS <- list(
  sample_col = "sample_id", condition_col = "condition", group_by = "cell_type_broad",
  contrasts = list(), min_cells_per_sample = 20L, min_replicates = 2L,
  method = "deseq2", padj = 0.05, lfc = 0.5
)
.S4_MIN_GENE_COUNTS <- 10
.S4_TOP_N <- 10L
.S4_DGE_COLS <- c("gene", "baseMean", "log2fc", "lfcSE", "pvalue", "padj", "direction")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

s4_config <- function(cfg) {
  d <- .scp_fill(cfg$dge %||% list(), .S4_DEFAULTS)
  if (is.null(d$contrasts)) d$contrasts <- list()
  p <- character()
  chk <- function(ok, msg) if (!isTRUE(ok)) p <<- c(p, msg)
  str1 <- function(v) is.character(v) && length(v) == 1 && nzchar(v)
  num1 <- function(v) is.numeric(v) && length(v) == 1 && !is.na(v)
  for (k in c("sample_col", "condition_col", "group_by")) chk(str1(d[[k]]), paste0("dge.", k, " must be a non-empty string"))
  chk(str1(d$method) && d$method %in% c("deseq2", "limma_voom"), "dge.method must be deseq2 or limma_voom")
  chk(num1(d$padj) && d$padj > 0 && d$padj <= 1, "dge.padj must be in (0, 1]")
  chk(num1(d$lfc) && d$lfc >= 0, "dge.lfc must be >= 0")
  chk(num1(d$min_cells_per_sample) && d$min_cells_per_sample >= 1, "dge.min_cells_per_sample must be >= 1")
  chk(num1(d$min_replicates) && d$min_replicates >= 1, "dge.min_replicates must be >= 1")
  chk(is.list(d$contrasts), "dge.contrasts must be a list")
  if (is.list(d$contrasts)) {
    for (i in seq_along(d$contrasts)) {
      ct <- d$contrasts[[i]]
      w <- paste0("dge.contrasts[", i, "]")
      if (!is.list(ct)) { chk(FALSE, paste0(w, " must be a mapping")); next }
      for (k in c("name", "numerator", "denominator")) chk(str1(ct[[k]]), paste0(w, ".", k, " must be a non-empty string"))
      if (str1(ct$name)) chk(grepl("^[A-Za-z0-9][A-Za-z0-9._-]*$", ct$name), paste0(w, ".name '", ct$name, "' has characters not allowed in file names"))
      if (!is.null(ct$covariates)) {
        cv <- unlist(ct$covariates)
        chk(is.character(cv) && all(nzchar(cv)), paste0(w, ".covariates must be a list of column names"))
        d$contrasts[[i]]$covariates <- as.character(cv)
      } else d$contrasts[[i]]$covariates <- character()
    }
    nm <- vapply(d$contrasts, function(ct) as.character(ct$name %||% ""), "")
    dup <- unique(nm[duplicated(nm) & nzchar(nm)])
    if (length(dup)) chk(FALSE, paste0("dge.contrasts: duplicated name(s) ", paste(dup, collapse = ", ")))
  }
  if (length(p)) scp_stop("invalid dge config:\n  ", paste(p, collapse = "\n  "))
  d$min_cells_per_sample <- as.integer(d$min_cells_per_sample)
  d$min_replicates <- as.integer(d$min_replicates)
  d
}

# --------------------------------------------------------------------------
# Labels (G3 decision -> S3/annotation.tsv -> cell_type_broad / cell_type_fine)
# --------------------------------------------------------------------------

.s4_chr <- function(x, default = NA_character_) {
  if (is.null(x) || length(x) == 0) return(default)
  v <- as.character(x[[1]])
  if (is.na(v)) default else v
}

#' Cluster table from gates/G3/decision.json ({"decision": {"clusters": [...]}}
#' or, tolerated, {"clusters": [...]}); columns cluster, label_broad,
#' label_fine, cl_id, confidence, status.
s4_annotation_from_decision <- function(path) {
  j <- tryCatch(jsonlite::fromJSON(path, simplifyVector = FALSE),
                error = function(e) scp_stop("cannot parse ", path, ": ", conditionMessage(e)))
  cl <- j$decision$clusters %||% j$clusters
  if (!is.list(cl) || !length(cl)) scp_stop(path, " has no decision.clusters entries")
  df <- do.call(rbind, lapply(cl, function(x) data.frame(
    cluster = .s4_chr(x$cluster), label_broad = .s4_chr(x$label_broad), label_fine = .s4_chr(x$label_fine),
    cl_id = .s4_chr(x$cl_id, ""), confidence = suppressWarnings(as.numeric(.s4_chr(x$confidence))),
    status = .s4_chr(x$status, ""), stringsAsFactors = FALSE)))
  if (any(is.na(df$cluster))) scp_stop(path, ": a cluster entry has no 'cluster'")
  if (anyDuplicated(df$cluster)) scp_stop(path, ": duplicated cluster(s) ", paste(unique(df$cluster[duplicated(df$cluster)]), collapse = ", "))
  df$label_broad[is.na(df$label_broad)] <- "unassigned"
  df$label_fine[is.na(df$label_fine)] <- df$label_broad[is.na(df$label_fine)]
  df
}

.s4_write_tsv <- function(df, path) {
  .scp_atomic(path, function(p) utils::write.table(df, p, sep = "\t", quote = FALSE, row.names = FALSE, na = ""))
}

#' Apply labels to the object. Returns list(obj, source, note, n_unassigned).
s4_apply_labels <- function(obj, run_dir) {
  dec <- file.path(run_dir, "gates", "G3", "decision.json")
  ann <- file.path(run_dir, "S3", "annotation.tsv")
  df <- NULL; source <- "none"; note <- NULL
  if (file.exists(dec)) {
    df <- s4_annotation_from_decision(dec)
    .s4_write_tsv(df, ann)
    source <- "gates/G3/decision.json"
    scp_log("labels: wrote S3/annotation.tsv from gates/G3/decision.json (", nrow(df), " clusters)")
  } else if (file.exists(ann)) {
    df <- scp_read_tsv(ann)
    need <- c("cluster", "label_broad")
    miss <- setdiff(need, colnames(df))
    if (length(miss)) scp_stop(ann, " is missing column(s): ", paste(miss, collapse = ", "))
    if (!("label_fine" %in% colnames(df))) df$label_fine <- df$label_broad
    source <- "S3/annotation.tsv"
    scp_log("labels: gates/G3/decision.json not found; using existing S3/annotation.tsv (", nrow(df), " clusters)")
  } else {
    note <- "no gates/G3/decision.json or S3/annotation.tsv: cell-type labels are not available"
    scp_warn_log("labels: ", note)
    return(list(obj = obj, source = source, note = note, n_unassigned = NA_integer_))
  }
  if (!("seurat_clusters" %in% colnames(obj[[]]))) scp_stop("the S2 object has no seurat_clusters column")
  cl <- as.character(obj$seurat_clusters)
  idx <- match(cl, as.character(df$cluster))
  n_un <- sum(is.na(idx))
  broad <- df$label_broad[idx]; fine <- df$label_fine[idx]
  broad[is.na(idx)] <- "unassigned"; fine[is.na(idx)] <- "unassigned"
  if (n_un > 0) {
    note <- sprintf("%d cell(s) in %d cluster(s) have no label in %s; labelled 'unassigned'", n_un,
                    length(unique(cl[is.na(idx)])), source)
    scp_warn_log("labels: ", note)
  }
  obj$cell_type_broad <- factor(broad)
  obj$cell_type_fine <- factor(fine)
  list(obj = obj, source = source, note = note, n_unassigned = as.integer(n_un))
}

# --------------------------------------------------------------------------
# Pseudobulk
# --------------------------------------------------------------------------

#' Sum counts per group x sample unit. Returns list(counts = dgCMatrix genes x
#' units, units = data.frame(pb_id, group, sample, n_cells), method).
s4_pseudobulk <- function(obj, group, sample, assay = "RNA") {
  group <- as.character(group); sample <- as.character(sample)
  if (length(group) != ncol(obj) || length(sample) != ncol(obj)) scp_stop("pseudobulk: group/sample vectors do not match the object")
  key <- paste(group, sample, sep = "\r")
  uk <- unique(key)
  units <- data.frame(pb_id = sprintf("pb%04d", seq_along(uk)),
                      group = group[match(uk, key)], sample = sample[match(uk, key)],
                      n_cells = as.integer(table(factor(key, levels = uk))), stringsAsFactors = FALSE)
  units <- units[order(units$group, units$sample), , drop = FALSE]
  units$pb_id <- sprintf("pb%04d", seq_len(nrow(units)))
  rownames(units) <- NULL
  idx <- match(key, paste(units$group, units$sample, sep = "\r"))
  pb_of_cell <- units$pb_id[idx]
  agg <- NULL; method <- "Seurat::AggregateExpression"
  if (nrow(units) > 1) {
    obj$scpipe_pb <- factor(pb_of_cell, levels = units$pb_id)
    agg <- tryCatch(suppressMessages(
      Seurat::AggregateExpression(obj, assays = assay, group.by = "scpipe_pb", layer = "counts", verbose = FALSE)[[assay]]),
      error = function(e) { scp_warn_log("AggregateExpression failed (", conditionMessage(e), "); summing counts directly"); NULL })
    if (!is.null(agg) && (!all(units$pb_id %in% colnames(agg)) || nrow(agg) != nrow(obj[[assay]]))) {
      scp_warn_log("AggregateExpression returned unexpected dimensions; summing counts directly")
      agg <- NULL
    }
  }
  if (is.null(agg)) {
    m <- scp_counts(obj, assay)
    ind <- Matrix::sparseMatrix(i = seq_len(ncol(m)), j = match(pb_of_cell, units$pb_id), x = 1,
                                dims = c(ncol(m), nrow(units)), dimnames = list(colnames(m), units$pb_id))
    agg <- m %*% ind
    method <- "direct sum"
  }
  agg <- scp_as_dgc(agg)[, units$pb_id, drop = FALSE]
  rownames(agg) <- rownames(obj[[assay]])
  list(counts = agg, units = units, method = method)
}

.s4_safe_name <- function(x) {
  s <- gsub("[^A-Za-z0-9.-]+", "-", as.character(x))
  s <- gsub("^-+|-+$", "", s)
  s[!nzchar(s)] <- "group"
  s
}

# --------------------------------------------------------------------------
# DGE methods
# --------------------------------------------------------------------------

#' Build the design for one group. Returns list(formula, covariates, notes,
#' ok, reason).
s4_design <- function(coldata, cond_col, covariates, min_replicates) {
  notes <- character()
  covs <- covariates
  # drop covariates with a single level in this group
  single <- covs[vapply(covs, function(cv) length(unique(coldata[[cv]])) < 2, TRUE)]
  if (length(single)) {
    notes <- c(notes, paste0("covariate(s) ", paste(single, collapse = ", "), " have one level in this group and were dropped"))
    covs <- setdiff(covs, single)
  }
  fml <- function(cv) stats::as.formula(paste("~", paste(c(cv, cond_col), collapse = " + ")))
  repeat {
    X <- stats::model.matrix(fml(covs), coldata)
    rank_ok <- qr(X)$rank == ncol(X)
    df_ok <- nrow(X) - ncol(X) >= 1
    if (rank_ok && df_ok) break
    if (!length(covs)) {
      return(list(ok = FALSE, reason = if (!rank_ok) "design is not full rank" else "no residual degrees of freedom",
                  covariates = covs, notes = notes, formula = fml(covs)))
    }
    dropped <- covs[length(covs)]
    notes <- c(notes, paste0("covariate ", dropped, " dropped: ", if (!rank_ok) "design not full rank" else "no residual degrees of freedom"))
    covs <- covs[-length(covs)]
  }
  list(ok = TRUE, reason = NULL, covariates = covs, notes = notes, formula = fml(covs))
}

s4_run_deseq2 <- function(counts, coldata, formula, cond_col, numerator, denominator, alpha) {
  if (!scp_has_pkg("DESeq2")) scp_stop("DESeq2 is not installed")
  storage.mode(counts) <- "integer"
  dds <- DESeq2::DESeqDataSetFromMatrix(countData = counts, colData = coldata, design = formula)
  dds <- suppressMessages(DESeq2::DESeq(dds, quiet = TRUE))
  res <- DESeq2::results(dds, contrast = c(cond_col, numerator, denominator), alpha = alpha)
  data.frame(gene = rownames(res), baseMean = as.numeric(res$baseMean), log2fc = as.numeric(res$log2FoldChange),
             lfcSE = as.numeric(res$lfcSE), pvalue = as.numeric(res$pvalue), padj = as.numeric(res$padj),
             stringsAsFactors = FALSE)
}

s4_run_limma_voom <- function(counts, coldata, formula, cond_col, numerator, denominator) {
  if (!scp_has_pkg("limma")) scp_stop("limma is not installed")
  lib <- colSums(counts)
  design <- stats::model.matrix(formula, coldata)
  coef <- paste0(cond_col, numerator)
  if (!(coef %in% colnames(design))) scp_stop("limma: coefficient ", coef, " not in the design (", paste(colnames(design), collapse = ", "), ")")
  v <- limma::voom(counts, design, lib.size = lib)
  fit <- limma::eBayes(limma::lmFit(v, design))
  tt <- limma::topTable(fit, coef = coef, number = Inf, sort.by = "none")
  lfcSE <- sqrt(fit$s2.post) * fit$stdev.unscaled[, coef]
  norm <- sweep(counts, 2, lib / mean(lib), "/")
  data.frame(gene = rownames(tt), baseMean = as.numeric(rowMeans(norm)[rownames(tt)]), log2fc = as.numeric(tt$logFC),
             lfcSE = as.numeric(lfcSE[rownames(tt)]), pvalue = as.numeric(tt$P.Value), padj = as.numeric(tt$adj.P.Val),
             stringsAsFactors = FALSE)
}

s4_direction <- function(log2fc, padj, padj_thr, lfc_thr) {
  d <- rep("ns", length(log2fc))
  sig <- !is.na(padj) & !is.na(log2fc) & padj < padj_thr
  d[sig & log2fc >= lfc_thr] <- "up"
  d[sig & log2fc <= -lfc_thr] <- "down"
  d
}

#' One contrast x group. Returns list(table (or NULL), method_used, notes, skipped_reason, n_tested).
s4_dge_group <- function(counts, coldata, cond_col, numerator, denominator, covariates, method, padj_thr, lfc_thr, min_replicates) {
  notes <- character()
  coldata[[cond_col]] <- factor(as.character(coldata[[cond_col]]), levels = c(denominator, numerator))
  for (cv in covariates) coldata[[cv]] <- factor(as.character(coldata[[cv]]))
  des <- s4_design(coldata, cond_col, covariates, min_replicates)
  notes <- c(notes, des$notes)
  if (!isTRUE(des$ok)) return(list(table = NULL, method_used = NULL, notes = notes, skipped_reason = des$reason, n_tested = 0L, covariates_used = des$covariates))
  keep <- Matrix::rowSums(counts) >= .S4_MIN_GENE_COUNTS
  m <- as.matrix(counts[keep, , drop = FALSE])
  n_tested <- nrow(m)
  if (n_tested < 2) return(list(table = NULL, method_used = NULL, notes = notes, skipped_reason = sprintf("only %d gene(s) with >= %d counts", n_tested, .S4_MIN_GENE_COUNTS), n_tested = n_tested, covariates_used = des$covariates))
  method_used <- method
  tab <- NULL
  if (method == "deseq2") {
    if (!scp_has_pkg("DESeq2")) {
      notes <- c(notes, "DESeq2 is not installed; limma_voom used instead")
      method_used <- "limma_voom"
    } else {
      tab <- tryCatch(s4_run_deseq2(m, coldata, des$formula, cond_col, numerator, denominator, padj_thr),
                      error = function(e) { notes <<- c(notes, paste0("DESeq2 failed (", conditionMessage(e), "); limma_voom used instead")); NULL })
      if (is.null(tab)) method_used <- "limma_voom"
    }
  }
  if (is.null(tab)) tab <- s4_run_limma_voom(m, coldata, des$formula, cond_col, numerator, denominator)
  tab$direction <- s4_direction(tab$log2fc, tab$padj, padj_thr, lfc_thr)
  o <- order(is.na(tab$padj), tab$padj, tab$pvalue, -abs(tab$log2fc))
  tab <- tab[o, .S4_DGE_COLS, drop = FALSE]
  rownames(tab) <- NULL
  list(table = tab, method_used = method_used, notes = notes, skipped_reason = NULL, n_tested = n_tested, covariates_used = des$covariates)
}

.s4_top <- function(tab, dir, n = .S4_TOP_N) {
  t <- tab[tab$direction == dir, , drop = FALSE]
  t <- t[order(t$padj, -abs(t$log2fc)), , drop = FALSE]
  t <- utils::head(t, n)
  lapply(seq_len(nrow(t)), function(i) list(gene = t$gene[i], log2fc = round(t$log2fc[i], 4), padj = signif(t$padj[i], 4)))
}

# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------

run_s4 <- function(cfg, repo) {
  out_dir <- scp_stage_dir(cfg, "S4")
  d <- s4_config(cfg)
  set.seed(cfg$seed %||% 42L)
  timings <- list()
  tm <- function(label, expr) { r <- scp_timed(label, expr); timings[[label]] <<- r$seconds; r$value }
  notes <- character()

  obj <- tm("load", scp_load_object(file.path(cfg$run_dir, "S2")))
  lab <- s4_apply_labels(obj, cfg$run_dir)
  obj <- lab$obj
  if (!is.null(lab$note)) notes <- c(notes, lab$note)
  md <- obj[[]]

  # gene map (for the Open Targets step and the report)
  fm <- tryCatch(obj[["RNA"]][[]], error = function(e) NULL)
  genes <- rownames(obj[["RNA"]])
  sym <- scp_symbols(obj)
  gid <- if (!is.null(fm) && "gene_id" %in% colnames(fm)) as.character(fm$gene_id) else rep("", length(genes))
  gid[is.na(gid)] <- ""
  ens <- ifelse(grepl("^ENSG[0-9]+", gid), sub("\\..*$", "", gid), ifelse(grepl("^ENSG[0-9]+", genes), sub("\\..*$", "", genes), ""))
  .s4_write_tsv(data.frame(gene = genes, symbol = sym, ensembl_id = ens, stringsAsFactors = FALSE), file.path(out_dir, "gene_map.tsv"))

  # grouping column
  gb_req <- d$group_by
  gb <- if (identical(gb_req, "cluster")) "seurat_clusters" else gb_req
  gb_note <- NULL
  if (!(gb %in% colnames(md))) {
    gb_note <- sprintf("group_by '%s' is not a cell metadata column%s; grouping by seurat_clusters instead", gb_req,
                       if (lab$source == "none") " (no cell-type labels available)" else "")
    scp_warn_log(gb_note); notes <- c(notes, gb_note)
    gb <- "seurat_clusters"
  }
  if (!(gb %in% colnames(md))) scp_stop("the S2 object has no '", gb, "' column")

  # per-cell sample / condition / covariates
  all_covs <- unique(unlist(lapply(d$contrasts, function(ct) ct$covariates)))
  need_cols <- unique(c(d$sample_col, d$condition_col, all_covs))
  miss <- setdiff(need_cols, colnames(md))
  if (length(miss)) scp_stop("dge: column(s) not in the cell metadata (sample sheet columns become cell metadata in S0): ",
                             paste(miss, collapse = ", "), "; available: ", paste(colnames(md), collapse = ", "))
  sample_v <- as.character(md[[d$sample_col]])
  group_v <- as.character(md[[gb]])
  unit_meta <- unique(data.frame(lapply(md[need_cols], as.character), stringsAsFactors = FALSE))
  if (anyDuplicated(unit_meta[[d$sample_col]])) {
    bad <- unique(unit_meta[[d$sample_col]][duplicated(unit_meta[[d$sample_col]])])
    scp_stop("dge: condition/covariates vary within sample_col '", d$sample_col, "' for ", paste(bad, collapse = ", "),
             " (each pseudobulk unit must have one condition and one value per covariate)")
  }
  rownames(unit_meta) <- unit_meta[[d$sample_col]]
  cond_levels <- sort(unique(unit_meta[[d$condition_col]]))
  for (ct in d$contrasts) for (k in c("numerator", "denominator"))
    if (!(ct[[k]] %in% cond_levels)) scp_stop("dge.contrasts '", ct$name, "': ", k, " '", ct[[k]], "' is not a value of '", d$condition_col,
                                               "' (values: ", paste(cond_levels, collapse = ", "), ")")

  pb <- tm("pseudobulk", s4_pseudobulk(obj, group_v, sample_v))
  units <- pb$units
  units$condition <- unit_meta[units$sample, d$condition_col]
  for (cv in all_covs) units[[cv]] <- unit_meta[units$sample, cv]
  scp_log(sprintf("pseudobulk: %d units (%d groups x %d samples) via %s", nrow(units), length(unique(units$group)),
                  length(unique(units$sample)), pb$method))
  if (!length(d$contrasts)) {
    notes <- c(notes, "dge.contrasts is empty: no differential expression was run")
    scp_warn_log("dge.contrasts is empty: nothing to test")
  }

  dge_dir <- file.path(out_dir, "dge"); dir.create(dge_dir, recursive = TRUE, showWarnings = FALSE)
  unlink(list.files(dge_dir, pattern = "\\.tsv\\.gz$", full.names = TRUE))
  method_used_all <- character()
  contrast_out <- list()
  for (ct in d$contrasts) {
    t0 <- proc.time()[["elapsed"]]
    scp_log("contrast ", ct$name, ": ", ct$numerator, " vs ", ct$denominator,
            if (length(ct$covariates)) paste0(" (covariates: ", paste(ct$covariates, collapse = ", "), ")") else "")
    u <- units[units$condition %in% c(ct$numerator, ct$denominator), , drop = FALSE]
    u$included <- u$n_cells >= d$min_cells_per_sample
    u$exclude_reason <- ifelse(u$included, "", sprintf("fewer than %d cells", d$min_cells_per_sample))
    pb_dir <- file.path(out_dir, "pseudobulk", ct$name); dir.create(pb_dir, recursive = TRUE, showWarnings = FALSE)
    inc <- u[u$included, , drop = FALSE]
    cm <- as.matrix(pb$counts[, inc$pb_id, drop = FALSE])
    scp_write_tsv_gz(data.frame(gene = rownames(cm), cm, check.names = FALSE, stringsAsFactors = FALSE), file.path(pb_dir, "counts.tsv.gz"))
    .s4_write_tsv(u, file.path(pb_dir, "samples.tsv"))
    samples_side <- list(numerator = sort(unique(inc$sample[inc$condition == ct$numerator])),
                         denominator = sort(unique(inc$sample[inc$condition == ct$denominator])))
    groups <- sort(unique(u$group))
    fnames <- make.unique(paste0(ct$name, "__", .s4_safe_name(groups)), sep = ".")
    group_out <- list()
    for (gi in seq_along(groups)) {
      g <- groups[[gi]]
      ug <- inc[inc$group == g, , drop = FALSE]
      n_num <- sum(ug$condition == ct$numerator); n_den <- sum(ug$condition == ct$denominator)
      cells <- list(numerator = sum(ug$n_cells[ug$condition == ct$numerator]), denominator = sum(ug$n_cells[ug$condition == ct$denominator]))
      entry <- list(group = g, file = NULL, n_samples = list(numerator = n_num, denominator = n_den), n_cells = cells,
                    covariates_used = ct$covariates, method_used = NULL, n_tested = 0L, n_up = 0L, n_down = 0L,
                    skipped_reason = NULL, notes = list(), top_up = list(), top_down = list())
      if (n_num < d$min_replicates || n_den < d$min_replicates) {
        entry$skipped_reason <- sprintf("%d %s and %d %s pseudobulk sample(s) with >= %d cells; min_replicates is %d per side",
                                        n_num, ct$numerator, n_den, ct$denominator, d$min_cells_per_sample, d$min_replicates)
        scp_log("  group ", g, ": skipped (", entry$skipped_reason, ")")
        group_out[[gi]] <- entry; next
      }
      coldata <- data.frame(ug[, c("sample", "condition", ct$covariates), drop = FALSE], row.names = ug$pb_id, stringsAsFactors = FALSE)
      colnames(coldata)[2] <- d$condition_col
      r <- s4_dge_group(pb$counts[, ug$pb_id, drop = FALSE], coldata, d$condition_col, ct$numerator, ct$denominator,
                        ct$covariates, d$method, d$padj, d$lfc, d$min_replicates)
      entry$notes <- as.list(r$notes); entry$covariates_used <- r$covariates_used; entry$n_tested <- as.integer(r$n_tested)
      if (is.null(r$table)) {
        entry$skipped_reason <- r$skipped_reason
        scp_log("  group ", g, ": skipped (", r$skipped_reason, ")")
        group_out[[gi]] <- entry; next
      }
      f <- file.path(dge_dir, paste0(fnames[[gi]], ".tsv.gz"))
      scp_write_tsv_gz(r$table, f)
      entry$file <- file.path("S4", "dge", basename(f))
      entry$method_used <- r$method_used
      method_used_all <- union(method_used_all, r$method_used)
      entry$n_up <- sum(r$table$direction == "up"); entry$n_down <- sum(r$table$direction == "down")
      entry$top_up <- .s4_top(r$table, "up"); entry$top_down <- .s4_top(r$table, "down")
      scp_log(sprintf("  group %s: %s, %d tested, %d up, %d down (%d vs %d samples)", g, r$method_used, entry$n_tested,
                      entry$n_up, entry$n_down, n_num, n_den))
      group_out[[gi]] <- entry
    }
    timings[[paste0("contrast_", ct$name)]] <- round(proc.time()[["elapsed"]] - t0, 3)
    contrast_out[[length(contrast_out) + 1]] <- list(
      name = ct$name, numerator = ct$numerator, denominator = ct$denominator, covariates = as.list(ct$covariates),
      samples = samples_side, pseudobulk_dir = file.path("S4", "pseudobulk", ct$name),
      n_units = nrow(u), n_units_excluded = sum(!u$included), groups = group_out)
  }

  summary <- list(
    stage = "S4", step = "dge", run_id = cfg$run_id, config_hash = cfg$config_hash, created_utc = .scp_utc(),
    method = list(requested = d$method, used = as.list(method_used_all),
                  available = list(DESeq2 = scp_has_pkg("DESeq2"), limma = scp_has_pkg("limma"))),
    group_by = list(requested = gb_req, used = gb, note = gb_note),
    labels = list(source = lab$source, note = lab$note, n_unassigned_cells = lab$n_unassigned),
    sample_col = d$sample_col, condition_col = d$condition_col, condition_levels = as.list(cond_levels),
    thresholds = list(padj = d$padj, lfc = d$lfc, min_cells_per_sample = d$min_cells_per_sample,
                      min_replicates = d$min_replicates, min_gene_counts = .S4_MIN_GENE_COUNTS),
    n_cells = ncol(obj), n_genes = nrow(obj), n_groups = length(unique(group_v)), n_samples = length(unique(sample_v)),
    pseudobulk_method = pb$method, gene_map = "S4/gene_map.tsv",
    contrasts = contrast_out, notes = as.list(notes),
    timings_s = timings, total_s = round(sum(unlist(timings)), 3),
    peak_memory = list(r_peak_rss_gb = scp_peak_mem_gb()),
    packages = list(R = paste(R.version$major, R.version$minor, sep = "."), Seurat = scp_pkg_version("Seurat"),
                    DESeq2 = scp_pkg_version("DESeq2"), limma = scp_pkg_version("limma"))
  )
  scp_write_json(summary, file.path(out_dir, "dge_summary.json"))
  n_tab <- sum(vapply(contrast_out, function(c) sum(vapply(c$groups, function(g) !is.null(g$file), TRUE)), 1))
  scp_log(sprintf("S4 DGE done: %d contrast(s), %d DGE table(s), group_by %s", length(contrast_out), n_tab, gb))
  invisible(summary)
}
