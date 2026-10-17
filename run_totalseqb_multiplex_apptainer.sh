#!/usr/bin/env bash
# TotalSeqB multiplex — 10x raw_feature_bc_matrix.h5 benchmark (Apptainer).
# Place the .h5 in the repo root or set INPUT. Results: bench/results/<RUN_NAME>/.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

H5_DEFAULT="${REPO}/60k_Human_TotalSeqB_hashing_3p_gem-x_Multiplex_count_raw_feature_bc_matrix.h5"

export INPUT="${INPUT:-${H5_DEFAULT}}"
export RUN_NAME="${RUN_NAME:-totalseqb_multiplex}"
export SIF_R_BENCH="${SIF_R_BENCH:-/scratch/${USER}/ddnb_r_full_sandbox}"
export SIF_GPU_BENCH="${SIF_GPU_BENCH:-/iridisfs/ddnb/images/rsc_scverse_cu12_sandbox}"
export BENCH_INSTALL_RAPIDS="${BENCH_INSTALL_RAPIDS:-0}"

# Do not inherit SKIP_PREP=1 from a PBMC3k shell session.
[[ "${SKIP_PREP+set}" == set ]] || unset SKIP_PREP

# Optional: ENGINES, SIZES, REPS, THREADS, TIMEOUT; SKIP_PREP=0 to force re-prep

exec bash "${REPO}/run_pbmc3k_apptainer.sh"
