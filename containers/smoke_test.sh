#!/usr/bin/env bash
# Smoke test of the scpipe image (run by .github/workflows/release.yml, and by hand after a pull):
#   docker run --rm --entrypoint bash ghcr.io/mukeshkbj/scpipe:latest /opt/scpipe/containers/smoke_test.sh
#   apptainer exec --nv scpipe.sif bash /opt/scpipe/containers/smoke_test.sh
# Preflight checks, then the whole pipeline (S0-S5) on the 500-cell test fixture with a mock LLM
# and offline Open Targets (tests/integration/run_full.sh). CPU only; a GPU is used if present.
set -euo pipefail
work="$(mktemp -d "${TMPDIR:-/tmp}/scpipe-smoke.XXXXXX")"
trap 'rm -rf "$work"' EXIT

scpipe doctor
python -c "import scfast; print('scfast', scfast.__version__, scfast.device_info())"
Rscript -e 'cat("scfastR", as.character(packageVersion("scfastR")), "\n")'
SCPIPE_DOCTOR_OFFLINE=1 PYTHON=python bash /opt/scpipe/tests/integration/run_full.sh "$work"
echo "smoke test passed"
