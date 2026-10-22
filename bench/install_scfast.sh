#!/usr/bin/env bash
# One-time: rustup + maturin develop scfast into BENCH_VENV (see docs/INSTALL.md).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export BENCH_VENV="${BENCH_VENV:-/scratch/${USER}/bench/venv}"
export BENCH_ROOT="${BENCH_ROOT:-/scratch/${USER}/bench}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_SHLVL
export PYTHONNOUSERSITE=1

module load gcc/13.3.0 2>/dev/null || module load gcc 2>/dev/null || true
py_mod="${BENCH_PYTHON_MODULE:-python/3.12.4}"
module load "${py_mod}" 2>/dev/null || module load python 2>/dev/null || true

# shellcheck source=bench/rustup_install.sh
source "$(dirname "$0")/rustup_install.sh"
bench_ensure_rustup

export REPO
log() { echo "[install_scfast] $*"; }
# shellcheck source=bench/apptainer_bench_lib.sh
source "${REPO}/bench/apptainer_bench_lib.sh"

if [[ ! -x "${BENCH_VENV}/bin/python3" ]]; then
  apptainer_bench_bootstrap_venv
fi

"${BENCH_VENV}/bin/python3" -m pip install -q maturin
bench_maturin_develop "${REPO}/rust/scfast-py"
"${BENCH_VENV}/bin/python3" -c "import scfast; print('ok scfast', scfast.__version__)"
