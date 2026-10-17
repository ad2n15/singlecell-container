#!/usr/bin/env bash
# PBMC 3k benchmark via Apptainer (10x filtered_gene_bc_matrices/hg19).
set -euo pipefail

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

cd /iridisfs/ddnb/Ahmed/singlecell

# Conda (base) breaks maturin/scfast build (VIRTUAL_ENV vs CONDA_PREFIX).
unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_SHLVL
export PYTHONNOUSERSITE=1

module load apptainer
# BENCH_PYTHON_MODULE unset => auto-pick a module python with _bz2 (see bench/apptainer_bench_lib.sh)
[[ -n "${BENCH_PYTHON_MODULE:-}" ]] && module load "${BENCH_PYTHON_MODULE}"
module load gocryptfs/2.4.0 2>/dev/null || module load gocryptfs 2>/dev/null || true
APPT_BIN="$(command -v apptainer)"

export SIF_PY_BENCH="${SIF_PY_BENCH:-/iridisfs/ddnb/images/celltuner_py_container_v2.sif}"
REPO="$(pwd)"
export REPO
# shellcheck source=bench/apptainer_bench_lib.sh
source "${REPO}/bench/apptainer_bench_lib.sh"
export SIF_GPU_BENCH="${SIF_GPU_BENCH:-$(apptainer_bench_rsc_sandbox)}"

export INPUT="${INPUT:-${REPO}/filtered_gene_bc_matrices/hg19}"
export BENCH_ROOT=/scratch/${USER}/bench/pbmc3k
export BENCH_VENV="${BENCH_VENV:-/scratch/${USER}/bench/venv}"
export WRAP="${BENCH_ROOT}/wrappers"
mkdir -p "${BENCH_ROOT}/logs"
LOG="${LOG_FILE:-${BENCH_ROOT}/logs/run_pbmc3k_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
set +o pipefail  # logging must not SIGPIPE-exit when tee backpressure (set -euo pipefail above)

log() { echo "[$(date -Iseconds)] $*"; }
log "log file: $LOG"
log "host: $(hostname)  user: $USER  pwd: $(pwd)"
apptainer_bench_ensure_r_image
apptainer_bench_ensure_gpu_image
apptainer_bench_init_apptainer_tmp
log "APPTAINER_TMPDIR=${APPTAINER_TMPDIR}"

read -r -a APPT_EXTRA <<< "${APPTAINER_EXTRA:---unsquash}"

for f in "$SIF_PY_BENCH"; do
  [[ -f "$f" ]] || { log "ERROR: missing image: $f"; exit 1; }
done
[[ -f "${SIF_R_BENCH}" || -d "${SIF_R_BENCH}" ]] || {
  log "ERROR: missing R image: ${SIF_R_BENCH} (expected $(apptainer_bench_r_sif))"
  exit 1
}
[[ -d "$INPUT" || -f "$INPUT" ]] || { log "ERROR: missing input (10x folder or .h5): $INPUT"; exit 1; }

log "input: $INPUT"
if [[ -n "${SIF_GPU_BENCH:-}" ]]; then
  log "images: seurat=$SIF_R_BENCH ($(apptainer_bench_image_kind "${SIF_R_BENCH}")) python=$SIF_PY_BENCH gpu=$SIF_GPU_BENCH ($(apptainer_bench_image_kind "${SIF_GPU_BENCH}"))"
else
  log "images: seurat=$SIF_R_BENCH ($(apptainer_bench_image_kind "${SIF_R_BENCH}")) python=$SIF_PY_BENCH gpu=none"
fi
log "apptainer: $APPT_BIN  extra: ${APPT_EXTRA[*]}"

if [[ -n "${ENGINES:-}" ]]; then
  log "NOTE: ENGINES from your shell (${ENGINES})"
else
  ENGINES="$(apptainer_bench_default_engines)"
  if apptainer_bench_has_gpu; then
    log "GPU detected; default ENGINES=seurat,scanpy-gpu,scfast-gpu (rapids via ${SIF_GPU_BENCH:-sandbox})"
  fi
fi
export ENGINES

apptainer_bench_default_install_rapids "${ENGINES}"
apptainer_bench_ensure_cuda_toolkit "${ENGINES}"
apptainer_bench_ensure_cugraph_lib "${ENGINES}"

apptainer_bench_bootstrap_venv
if apptainer_bench_engines_need_scfast "${ENGINES}"; then
  apptainer_bench_bootstrap_scfast
else
  log "skip scfast Rust build (ENGINES has no scfast-cpu/scfast-gpu)"
fi
apptainer_bench_bootstrap_rapids
apptainer_bench_bootstrap_leidenbase
log "bootstrap: leidenbase ok"
apptainer_bench_write_wrappers
log "bootstrap: wrappers ok"

export PATH="${WRAP}:${PATH}"
export RSCRIPT="${WRAP}/Rscript"
export PYTHON="${WRAP}/python3"
export RAPIDS_PYTHON="${BENCH_RAPIDS_PYTHON:-${WRAP}/python3-gpu}"
[[ -x "${RAPIDS_PYTHON}" ]] || export RAPIDS_PYTHON="${WRAP}/python3"
RUN_NAME="${RUN_NAME:-pbmc3k}"
_prep_data="${REPO}/bench/results/${RUN_NAME}/data"
_prep_ok=0
if [[ -d "${_prep_data}" ]] && compgen -G "${_prep_data}/n*" >/dev/null; then
  _prep_ok=1
fi
if [[ "${SKIP_PREP:-}" == "1" ]]; then
  if [[ "${_prep_ok}" == "1" ]]; then
    log "SKIP_PREP=1 (prep present under ${_prep_data})"
  else
    log "WARN: SKIP_PREP=1 but no ${_prep_data}/n*; running prep"
    unset SKIP_PREP
  fi
elif [[ "${SKIP_PREP:-}" == "0" ]]; then
  log "SKIP_PREP=0; re-running prep"
elif [[ "${_prep_ok}" == "1" ]]; then
  export SKIP_PREP=1
  log "prep data exists under ${_prep_data}; SKIP_PREP=1 (set SKIP_PREP=0 to re-run prep)"
fi

apptainer_bench_require_gpu_python "${ENGINES}" || exit 1
apptainer_bench_require_scfast_gpu "${ENGINES}" || exit 1

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
