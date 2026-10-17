#!/usr/bin/env bash
# OneK1K benchmark via Apptainer (no host Seurat/scanpy install).
# Images under /iridisfs/ddnb/images/ (override with env vars below).
set -euo pipefail

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

cd /iridisfs/ddnb/Ahmed/singlecell

module load apptainer
[[ -n "${BENCH_PYTHON_MODULE:-}" ]] && module load "${BENCH_PYTHON_MODULE}"
module load gocryptfs/2.4.0 2>/dev/null || module load gocryptfs 2>/dev/null || true
APPT_BIN="$(command -v apptainer)"

# SIF images (override if needed)
export SIF_R_EXPORT="${SIF_R_EXPORT:-/iridisfs/ddnb/images/ddnb_r_full.sif}"
export SIF_R_BENCH="${SIF_R_BENCH:-/iridisfs/ddnb/images/ddnb_r_full.sif}"
export SIF_PY_BENCH="${SIF_PY_BENCH:-/iridisfs/ddnb/images/celltuner_py_container_v2.sif}"
# Optional GPU engine (needs rapids-singlecell inside the image; often off)
export SIF_GPU="${SIF_GPU:-/iridisfs/ddnb/images/bionemo-framework_nightly.sif}"

export RDS=/iridisfs/ddnb/data/Onek1k/b3e8792e-a31b-404b-a866-250c43bc06d5.rds
export BENCH_ROOT=/scratch/${USER}/bench/onek1k_b3e8792e
export TENX="${BENCH_ROOT}/counts_10x"
export LABELS="${BENCH_ROOT}/labels.tsv"
export COUNTS_CKPT="${BENCH_ROOT}/counts_matrix.rds"
# Skip RDS export: set BENCH_INPUT to .h5ad (same OneK1K cohort) or existing 10x folder.
export BENCH_INPUT="${BENCH_INPUT:-}"
mkdir -p "${BENCH_ROOT}/logs"
LOG="${LOG_FILE:-${BENCH_ROOT}/logs/run_1m_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date -Iseconds)] $*"; }
log "log file: $LOG"
log "host: $(hostname)  user: $USER  pwd: $(pwd)"

REPO="$(pwd)"
# shellcheck source=bench/apptainer_bench_lib.sh
source "${REPO}/bench/apptainer_bench_lib.sh"
export BENCH_VENV="${BENCH_VENV:-/scratch/${USER}/bench/venv}"

read -r -a APPT_EXTRA <<< "${APPTAINER_EXTRA:---unsquash}"

BIND=(
  --bind "${REPO}:${REPO}"
  --bind /iridisfs/ddnb:/iridisfs/ddnb
  --bind /scratch:/scratch
)

appt_r() {
  local img="$1"; shift
  "$APPT_BIN" exec "${APPT_EXTRA[@]}" "${BIND[@]}" "$img" "$@"
}

for f in "$SIF_R_EXPORT" "$SIF_R_BENCH" "$SIF_PY_BENCH"; do
  [[ -f "$f" ]] || { log "ERROR: missing image: $f"; exit 1; }
done
log "images: export=$SIF_R_EXPORT seurat=$SIF_R_BENCH python=$SIF_PY_BENCH"
log "apptainer: $APPT_BIN  extra: ${APPT_EXTRA[*]}"

BENCH_DATA="${TENX}"
if [[ -n "${BENCH_INPUT}" ]]; then
  BENCH_DATA="${BENCH_INPUT}"
  log "skip export (BENCH_INPUT=${BENCH_INPUT})"
elif [[ -f "${TENX}/matrix.mtx.gz" || -f "${TENX}/matrix.mtx" || -f "${TENX}/matrix.h5" ]]; then
  log "skip export (10x already at ${TENX})"
elif [[ -f "${COUNTS_CKPT}" ]]; then
  log "resume export: counts checkpoint only (no RDS re-read) -> ${TENX}"
  rm -rf "${TENX}"
  export TENX LABELS COUNTS_CKPT
  appt_r "$SIF_R_EXPORT" Rscript - <<'RS'
suppressPackageStartupMessages(library(DropletUtils))
tenx <- Sys.getenv("TENX")
ckpt <- Sys.getenv("COUNTS_CKPT")
counts <- readRDS(ckpt)
message("[export] loaded checkpoint ", ckpt, " ", ncol(counts), "x", nrow(counts))
tenx_tmp <- paste0(tenx, ".writing")
if (dir.exists(tenx_tmp)) unlink(tenx_tmp, recursive = TRUE)
if (dir.exists(tenx)) unlink(tenx, recursive = TRUE)
message("[export] writing 10x to ", tenx_tmp, " ...")
DropletUtils::write10xCounts(
  tenx_tmp, counts,
  barcodes = colnames(counts),
  gene.id = rownames(counts),
  gene.type = "Gene Expression",
  version = "3"
)
file.rename(tenx_tmp, tenx)
message("[export] done -> ", tenx)
RS
  log "END export (from checkpoint)"
else
  rm -rf "${TENX}" "${TENX}.writing"
  log "START export RDS -> 10x (${SIF_R_EXPORT})"
  log "RDS=${RDS}  (next failed run can resume from ${COUNTS_CKPT} without re-reading RDS)"
  export TENX LABELS RDS COUNTS_CKPT
  appt_r "$SIF_R_EXPORT" Rscript - <<'RS'
suppressPackageStartupMessages({
  library(Seurat)
  library(DropletUtils)
})
rds <- Sys.getenv("RDS")
tenx <- Sys.getenv("TENX")
labels <- Sys.getenv("LABELS")
ckpt <- Sys.getenv("COUNTS_CKPT")
if (file.exists(ckpt)) {
  message("[export] loading counts checkpoint ", ckpt)
  counts <- readRDS(ckpt)
} else {
  message("[export] reading RDS: ", rds)
  obj <- readRDS(rds)
  message("[export] loaded Seurat object, cells=", ncol(obj), " features=", nrow(obj))
  if (!file.exists(labels)) {
    meta <- obj[[]]
    ct <- if ("predicted.celltype.l2" %in% colnames(meta)) meta$predicted.celltype.l2 else Idents(obj)
    write.table(
      data.frame(barcode = Cells(obj), cell_type = as.character(ct)),
      labels, sep = "\t", row.names = FALSE, quote = FALSE
    )
    message("[export] wrote labels: ", labels)
  } else {
    message("[export] labels exist, skip: ", labels)
  }
  assay <- DefaultAssay(obj)
  counts <- tryCatch(
    LayerData(obj, assay = assay, layer = "counts"),
    error = function(e) tryCatch(
      GetAssayData(obj, assay = assay, slot = "counts"),
      error = function(e2) GetAssayData(obj, assay = assay, layer = "counts")
    )
  )
  if (is.null(counts) || ncol(counts) == 0) {
    counts <- tryCatch(
      LayerData(obj, assay = assay, layer = "data"),
      error = function(e) GetAssayData(obj, assay = assay, slot = "data")
    )
  }
  message("[export] saving counts checkpoint ", ckpt, " ...")
  saveRDS(counts, ckpt)
  message("[export] checkpoint saved")
}
tenx_tmp <- paste0(tenx, ".writing")
if (dir.exists(tenx_tmp)) unlink(tenx_tmp, recursive = TRUE)
if (dir.exists(tenx)) unlink(tenx, recursive = TRUE)
message("[export] writing 10x to ", tenx_tmp, " ...")
DropletUtils::write10xCounts(
  tenx_tmp, counts,
  barcodes = colnames(counts),
  gene.id = rownames(counts),
  gene.type = "Gene Expression",
  version = "3"
)
file.rename(tenx_tmp, tenx)
message("[export] done: cells=", ncol(counts), " -> ", tenx)
RS
  log "END export RDS -> 10x"
fi
BENCH_DATA="${BENCH_DATA:-${TENX}}"

export WRAP="${BENCH_ROOT}/wrappers"
apptainer_bench_bootstrap_venv
apptainer_bench_bootstrap_scfast
apptainer_bench_bootstrap_leidenbase
apptainer_bench_write_wrappers

export PATH="${WRAP}:${PATH}"
export RSCRIPT="${WRAP}/Rscript"
export PYTHON="${WRAP}/python3"

# rapids only if you set ENGINES including rapids AND the GPU image has rapids-singlecell
ENGINES="${ENGINES:-seurat,scanpy,scfast-cpu}"
log "START benchmark engines=${ENGINES} sizes=${SIZES:-10000,50000,100000}"
log "bench output: ${REPO}/bench/results/onek1k_b3e8792e/"

if [[ -d "${REPO}/bench/results/onek1k_b3e8792e/data/n10000" ]]; then
  export SKIP_PREP=1
  log "prep data exists; SKIP_PREP=1 (won't re-read 1.2M h5ad)"
fi

nice -n 10 bash bench/run_bench.sh \
  --input "${BENCH_DATA}" \
  --name onek1k_b3e8792e \
  --sizes "${SIZES:-10000,50000,100000}" \
  --engines "${ENGINES}" \
  --labels "${LABELS}" \
  --reps 3 \
  --threads 8 \
  --timeout 14400 \
  --qc "--min-genes 200 --max-genes 5000 --max-mt 25"

log "END benchmark"
log "Report: ${REPO}/bench/results/onek1k_b3e8792e/report.md"
