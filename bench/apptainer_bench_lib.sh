# Shared Apptainer helpers for run_*_apptainer.sh (source, do not execute).
apptainer_bench_bind=(
  --bind "${REPO}:${REPO}"
  --bind /iridisfs/ddnb:/iridisfs/ddnb
  --bind /scratch:/scratch
  --bind /iridisfs/ixsoftware:/iridisfs/ixsoftware
)

apptainer_bench_refresh_venv_meta() {
  [[ -f "${BENCH_VENV}/bin/python3" ]] || return 1
  local host_py site_pkgs
  host_py="$(readlink -f "${BENCH_VENV}/bin/python3")"
  site_pkgs="$(ls -d "${BENCH_VENV}"/lib/python*/site-packages 2>/dev/null | head -1)"
  printf '%s\n' "${host_py}" > "${BENCH_VENV}/host_python.txt"
  printf '%s\n' "${site_pkgs}" > "${BENCH_VENV}/site_packages.txt"
}

# Host Python venv on scratch (prep, Scanpy, and Seurat reticulate/Leiden).
apptainer_bench_bootstrap_venv() {
  local stamp="${BENCH_VENV}/.bootstrap_ok"
  if [[ -f "${BENCH_VENV}/bin/python3" && -f "$stamp" ]]; then
    apptainer_bench_refresh_venv_meta
    return 0
  fi

  # python/3.12.6 on Iridis X is missing _bz2; scanpy/anndata/zarr need it.
  local py_mod host_py
  if [[ -n "${BENCH_PYTHON_MODULE:-}" ]]; then
    module load "${BENCH_PYTHON_MODULE}"
    py_mod="${BENCH_PYTHON_MODULE}"
  else
    py_mod=""
    for cand in python/3.12.4 python/3.13.1 python/3.12.6 python/3.14.0; do
      module load "$cand" 2>/dev/null || continue
      if python3 -c "import bz2" 2>/dev/null; then
        py_mod="$cand"
        break
      fi
    done
    [[ -n "$py_mod" ]] || { log "ERROR: no module python with bz2 (_bz2); set BENCH_PYTHON_MODULE"; exit 1; }
  fi
  # Conda base + PYTHONPATH confuse venv/pip on login nodes; use module Python only.
  unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV
  export PYTHONNOUSERSITE=1
  host_py="$(command -v python3)"
  log "bootstrap venv -> ${BENCH_VENV} (module ${py_mod}, ${host_py})"
  mkdir -p "${BENCH_ROOT}/logs"
  if [[ -d "${BENCH_VENV}" && ! -f "$stamp" ]]; then
    log "removing incomplete venv ${BENCH_VENV}"
    rm -rf "${BENCH_VENV}"
  fi
  "${host_py}" -m venv --clear "${BENCH_VENV}"
  "${BENCH_VENV}/bin/python3" -m pip install -U pip wheel 2>&1 | tee -a "${BENCH_ROOT}/logs/bootstrap_venv.log"
  "${BENCH_VENV}/bin/python3" -m pip install --ignore-installed \
    scanpy anndata h5py leidenalg python-igraph scikit-misc 2>&1 \
    | tee -a "${BENCH_ROOT}/logs/bootstrap_venv.log"
  "${BENCH_VENV}/bin/python3" - <<'PY' | tee -a "${BENCH_ROOT}/logs/bootstrap_venv.log"
import scanpy, leidenalg, igraph, skmisc
for m in (scanpy, leidenalg, igraph, skmisc):
    print("ok", m.__name__, m.__file__)
PY
  BENCH_HOST_PYTHON="${host_py}"
  BENCH_SITE_PKGS="$("${BENCH_VENV}/bin/python3" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  printf '%s\n' "${BENCH_HOST_PYTHON}" > "${BENCH_VENV}/host_python.txt"
  printf '%s\n' "${BENCH_SITE_PKGS}" > "${BENCH_VENV}/site_packages.txt"
  touch "$stamp"
}

# scfast Rust engine (bench/run_py.py --engine scfast-cpu); optional if rust/maturin missing.
apptainer_bench_bootstrap_scfast() {
  if "${BENCH_VENV}/bin/python3" -c "import scfast" 2>/dev/null; then
    log "scfast already importable in ${BENCH_VENV}"
    return 0
  fi
  [[ "${BENCH_BUILD_SCFAST:-1}" == "1" ]] || { log "scfast not installed (BENCH_BUILD_SCFAST=0)"; return 0; }
  log "building scfast into ${BENCH_VENV} (needs rust + gcc modules; set BENCH_BUILD_SCFAST=0 to skip)"
  mkdir -p "${BENCH_ROOT}/logs"
  module load rust 2>/dev/null || true
  module load gcc/13.3.0 2>/dev/null || module load gcc 2>/dev/null || true
  if ! command -v cargo >/dev/null 2>&1; then
    log "WARN: cargo not found; scfast-cpu will be unavailable in the report"
    return 0
  fi
  if ! "${BENCH_VENV}/bin/python3" -m pip install -q maturin 2>&1 | tee -a "${BENCH_ROOT}/logs/bootstrap_scfast.log"; then
    log "WARN: maturin install failed"
    return 0
  fi
  (
    cd "${REPO}/rust/scfast-py"
    "${BENCH_VENV}/bin/maturin" develop --release
  ) 2>&1 | tee -a "${BENCH_ROOT}/logs/bootstrap_scfast.log" || {
    log "WARN: scfast build failed; see ${BENCH_ROOT}/logs/bootstrap_scfast.log"
    return 0
  }
  "${BENCH_VENV}/bin/python3" -c "import scfast; print('scfast', scfast.__version__)"
}

# leidenbase for Seurat >= 5.1 (avoids reticulate when available).
apptainer_bench_bootstrap_leidenbase() {
  local stamp="${BENCH_ROOT}/.leidenbase_ok_${SIF_R_BENCH##*/}"
  [[ -f "$stamp" ]] && return 0
  log "bootstrap R leidenbase (image ${SIF_R_BENCH})"
  mkdir -p "${BENCH_ROOT}/logs"
  "${APPT_BIN}" exec "${APPT_EXTRA[@]}" "${apptainer_bench_bind[@]}" "${SIF_R_BENCH}" \
    Rscript -e '
      if (!requireNamespace("leidenbase", quietly = TRUE)) {
        if (!requireNamespace("BiocManager", quietly = TRUE))
          install.packages("BiocManager", repos = "https://cloud.r-project.org")
        BiocManager::install("leidenbase", ask = FALSE, update = FALSE)
      }
      stopifnot(requireNamespace("leidenbase", quietly = TRUE))
      message("leidenbase ", as.character(packageVersion("leidenbase")))
    ' | tee -a "${BENCH_ROOT}/logs/bootstrap_leidenbase.log"
  touch "$stamp"
}

apptainer_bench_write_wrappers() {
  mkdir -p "${WRAP}"
  cat > "${WRAP}/Rscript" <<EOF
#!/usr/bin/env bash
# Clustering uses leidenbase in bench/run_seurat.R (no reticulate/leidenalg in SIF).
exec ${APPT_BIN} exec ${APPT_EXTRA[*]} \\
  --bind ${REPO}:${REPO} --bind /iridisfs/ddnb:/iridisfs/ddnb --bind /scratch:/scratch \\
  --bind /iridisfs/ixsoftware:/iridisfs/ixsoftware \\
  ${SIF_R_BENCH} \\
  env -u VIRTUAL_ENV -u RETICULATE_PYTHON -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \\
  Rscript "\$@"
EOF
  cat > "${WRAP}/python3" <<EOF
#!/usr/bin/env bash
exec ${BENCH_VENV}/bin/python3 "\$@"
EOF
  chmod +x "${WRAP}/Rscript" "${WRAP}/python3"
  log "wrappers written under ${WRAP}"
}
