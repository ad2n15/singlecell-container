# R packages for every scpipe stage (R/stages/*.R), installed by containers/Dockerfile.
# Also usable on a native install: Rscript containers/install_r_packages.R
# (binaries come from the Posit Package Manager when the repos option points there).

cran <- c(
  # S0-S2 and the exchange format
  "Seurat", "SeuratObject", "Matrix", "hdf5r", "jsonlite", "digest", "yaml", "optparse",
  "data.table", "R.utils", "future", "irlba", "igraph", "RhpcBLASctl", "qs2",
  # S2 core: integration, clustering, embedding
  "harmony", "leidenbase", "uwot", "glmGamPoi",
  # tooling
  "remotes", "BiocManager", "testthat"
)
bioc <- c("DESeq2", "limma", "edgeR", "SummarizedExperiment", "scDblFinder", "BiocParallel")
github <- c("bnprks/BPCells/r", "immunogenomics/presto")

install.packages(setdiff(cran, rownames(installed.packages())))
BiocManager::install(bioc, update = FALSE, ask = FALSE)
remotes::install_github(github, upgrade = "never")

need <- c("Seurat", "SeuratObject", "hdf5r", "harmony", "leidenbase", "BPCells", "presto",
          "DESeq2", "limma", "qs2")
missing <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("R packages failed to install: ", paste(missing, collapse = ", "))
