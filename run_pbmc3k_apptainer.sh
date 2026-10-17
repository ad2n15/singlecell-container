#!/usr/bin/env bash
# PBMC 3k benchmark via Apptainer (10x filtered_gene_bc_matrices/hg19).
set -euo pipefail

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

cd /iridisfs/ddnb/Ahmed/singlecell

module load apptainer
# BENCH_PYTHON_MODULE unset => auto-pick a module python with _bz2 (see bench/apptainer_bench_lib.sh)
[[ -n "${BENCH_PYTHON_MODULE:-}" ]] && module load "${BENCH_PYTHON_MODULE}"
module load gocryptfs/2.4.0 2>/dev/null || module load gocryptfs 2>/dev/null || true
APPT_BIN="$(command -v apptainer)"

export SIF_R_BENCH="${SIF_R_BENCH:-/iridisfs/ddnb/images/ddnb_r_full.sif}"
export SIF_PY_BENCH="${SIF_PY_BENCH:-/iridisfs/ddnb/images/celltuner_py_container_v2.sif}"

REPO="$(pwd)"
# shellcheck source=bench/apptainer_bench_lib.sh
source "${REPO}/bench/apptainer_bench_lib.sh"

export INPUT="${INPUT:-${REPO}/filtered_gene_bc_matrices/hg19}"
export BENCH_ROOT=/scratch/${USER}/bench/pbmc3k
export BENCH_VENV="${BENCH_VENV:-/scratch/${USER}/bench/venv}"
export WRAP="${BENCH_ROOT}/wrappers"
mkdir -p "${BENCH_ROOT}/logs"
LOG="${LOG_FILE:-${BENCH_ROOT}/logs/run_pbmc3k_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date -Iseconds)] $*"; }
log "log file: $LOG"
log "host: $(hostname)  user: $USER  pwd: $(pwd)"

read -r -a APPT_EXTRA <<< "${APPTAINER_EXTRA:---unsquash}"

for f in "$SIF_R_BENCH" "$SIF_PY_BENCH"; do
  [[ -f "$f" ]] || { log "ERROR: missing image: $f"; exit 1; }
done
[[ -d "$INPUT" ]] || { log "ERROR: missing 10x folder: $INPUT (run Rscript seurat_pbmc3k.R to download)"; exit 1; }

log "input: $INPUT"
log "images: seurat=$SIF_R_BENCH python=$SIF_PY_BENCH"
log "apptainer: $APPT_BIN  extra: ${APPT_EXTRA[*]}"

apptainer_bench_bootstrap_venv
apptainer_bench_bootstrap_scfast
apptainer_bench_bootstrap_leidenbase
apptainer_bench_write_wrappers

export PATH="${WRAP}:${PATH}"
export RSCRIPT="${WRAP}/Rscript"
export PYTHON="${WRAP}/python3"

ENGINES="${ENGINES:-seurat,scanpy,scfast-cpu}"
RUN_NAME="${RUN_NAME:-pbmc3k}"
if [[ -d "${REPO}/bench/results/${RUN_NAME:-pbmc3k}/data/n2638" ]]; then
  export SKIP_PREP=1
  log "prep data exists; SKIP_PREP=1"
fi

log "START benchmark name=${RUN_NAME} engines=${ENGINES} sizes=${SIZES:-all}"
log "bench output: ${REPO}/bench/results/${RUN_NAME}/"

nice -n 10 bash bench/run_bench.sh \
  --input "${INPUT}" \
  --name "${RUN_NAME}" \
  --sizes "${SIZES:-all}" \
  --engines "${ENGINES}" \
  --reps "${REPS:-3}" \
  --threads "${THREADS:-8}" \
  --timeout "${TIMEOUT:-7200}"

log "END benchmark"
log "Report: ${REPO}/bench/results/${RUN_NAME}/report.md"
