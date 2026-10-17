#!/usr/bin/env bash
# OneK1K ~1M-cell benchmark (Seurat vs Scanpy). Run on Iridis X login node.
set -euo pipefail

cd /iridisfs/ddnb/Ahmed/singlecell

module load R/4.5.1-mkl

export RDS=/iridisfs/ddnb/data/Onek1k/b3e8792e-a31b-404b-a866-250c43bc06d5.rds
export BENCH_ROOT=/scratch/${USER}/bench/onek1k_b3e8792e
export TENX="${BENCH_ROOT}/counts_10x"
export LABELS="${BENCH_ROOT}/labels.tsv"
mkdir -p "${BENCH_ROOT}"

# Python for bench/prep.py (create once: ~2 min)
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q scanpy scikit-misc leidenalg python-igraph h5py anndata pandas scipy numpy matplotlib
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# One-time R setup (slow; skip after Seurat + DropletUtils are installed)
if ! Rscript -e 'stopifnot(requireNamespace("Seurat", quietly=TRUE), requireNamespace("DropletUtils", quietly=TRUE))' 2>/dev/null; then
  echo "Installing Seurat + DropletUtils into your user R library (one-time)..."
  Rscript -e 'if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager", repos="https://cloud.r-project.org"); BiocManager::install(c("DropletUtils"), ask=FALSE, update=FALSE); install.packages("Seurat", repos="https://cloud.r-project.org")'
fi

if [[ ! -f "${TENX}/matrix.mtx.gz" && ! -f "${TENX}/matrix.mtx" ]]; then
  echo "== Export RDS -> 10x (shared input for prep; may take a while, needs ~64GB+ RAM)"
  export TENX LABELS RDS
  Rscript - <<'RS'
suppressPackageStartupMessages({
  library(Seurat)
  library(DropletUtils)
})
rds <- Sys.getenv("RDS")
tenx <- Sys.getenv("TENX")
labels <- Sys.getenv("LABELS")
message("reading ", rds)
obj <- readRDS(rds)
meta <- obj[[]]
ct <- if ("predicted.celltype.l2" %in% colnames(meta)) meta$predicted.celltype.l2 else Idents(obj)
write.table(
  data.frame(barcode = Cells(obj), cell_type = as.character(ct)),
  labels, sep = "\t", row.names = FALSE, quote = FALSE
)
counts <- GetAssayData(obj, assay = DefaultAssay(obj), layer = "counts")
if (is.null(counts) || ncol(counts) == 0) counts <- GetAssayData(obj, assay = DefaultAssay(obj), layer = "data")
dir.create(tenx, recursive = TRUE, showWarnings = FALSE)
DropletUtils::write10xCounts(
  tenx, counts,
  barcodes = colnames(counts),
  gene.id = rownames(counts),
  gene.type = "Gene Expression",
  version = "3"
)
message("cells: ", ncol(counts), " -> ", tenx)
RS
fi

nice -n 10 bash bench/run_bench.sh \
  --input "${TENX}" \
  --name onek1k_b3e8792e \
  --sizes 10000,50000,100000 \
  --engines seurat,scanpy,rapids \
  --labels "${LABELS}" \
  --reps 3 \
  --threads 8 \
  --timeout 14400 \
  --qc "--min-genes 200 --max-genes 5000 --max-mt 25"

echo "Report: $(pwd)/bench/results/onek1k_b3e8792e/report.md"
