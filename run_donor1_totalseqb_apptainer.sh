#!/usr/bin/env bash
# Donor1 PBMC TotalSeqB — 10x filtered_feature_bc_matrix.h5 benchmark (Apptainer).
# Same engines/images as run_pbmc3k_apptainer.sh; results under bench/results/<RUN_NAME>/.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

H5_DEFAULT="${REPO}/60k_Human_TotalSeqB_hashing_3p_gem-x_Human_Donor1_PBMC_TotalSeqB_hashing_3p_gem-x_count_sample_filtered_feature_bc_matrix.h5"

export INPUT="${INPUT:-${H5_DEFAULT}}"
export RUN_NAME="${RUN_NAME:-donor1_totalseqb}"
export SIF_R_BENCH="${SIF_R_BENCH:-/scratch/${USER}/ddnb_r_full_sandbox}"
export SIF_GPU_BENCH="${SIF_GPU_BENCH:-/iridisfs/ddnb/images/rsc_scverse_cu12_sandbox}"
export BENCH_INSTALL_RAPIDS="${BENCH_INSTALL_RAPIDS:-0}"

[[ "${SKIP_PREP+set}" == set ]] || unset SKIP_PREP

# Optional: ENGINES, SIZES, REPS, THREADS, TIMEOUT; SKIP_PREP=0 to force re-prep

exec bash "${REPO}/run_pbmc3k_apptainer.sh"
