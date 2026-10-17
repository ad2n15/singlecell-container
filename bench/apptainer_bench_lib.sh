# Shared Apptainer helpers for run_*_apptainer.sh (source, do not execute).
# rapids-singlecell GPU bench (Apptainer sandbox from ghcr.io/scverse/rapids-singlecell-cu12:latest).
apptainer_bench_rsc_sandbox() {
  printf '%s\n' "/iridisfs/ddnb/images/rsc_scverse_cu12_sandbox"
}

# Seurat/R bench: prefer sandbox (avoids --unsquash permission/fuse issues on Iridis X).
apptainer_bench_r_sif() {
  printf '%s\n' "/iridisfs/ddnb/images/ddnb_r_full.sif"
}

# ddnb/images sandboxes often cannot be executed (permission denied); use scratch per-user.
apptainer_bench_r_sandbox() {
  printf '%s\n' "/scratch/${USER}/ddnb_r_full_sandbox"
}

apptainer_bench_r_sandbox_shared() {
  printf '%s\n' "/iridisfs/ddnb/images/ddnb_r_full_sandbox"
}

apptainer_bench_resolve_r_sandbox() {
  local c
  for c in "$(apptainer_bench_r_sandbox)" "$(apptainer_bench_r_sandbox_shared)"; do
    [[ -d "${c}" ]] && { printf '%s\n' "${c}"; return 0; }
  done
  return 1
}

apptainer_bench_r_build_cmd() {
  printf '%s\n' "apptainer build --sandbox $(apptainer_bench_r_sandbox) $(apptainer_bench_r_sif)"
}

# R/Seurat on Iridis X: .sif needs --unsquash (often "permission denied" on login nodes). Same bits as:
#   apptainer build --sandbox <sand> <sif>  then exec the sandbox (no unsquash).
# Auto-pick sandbox when present unless BENCH_R_FORCE_SIF=1.
apptainer_bench_ensure_r_image() {
  local sif sand
  sif="$(apptainer_bench_r_sif)"
  if [[ -n "${SIF_R_BENCH:-}" && ( -f "${SIF_R_BENCH}" || -d "${SIF_R_BENCH}" ) ]]; then
    log "R bench image: ${SIF_R_BENCH} ($(apptainer_bench_image_kind "${SIF_R_BENCH}"))"
    if [[ -d "${SIF_R_BENCH}" && "${SIF_R_BENCH}" == "$(apptainer_bench_r_sandbox_shared)" ]]; then
      log "WARN: ${SIF_R_BENCH} is often not runnable (permission denied); use $(apptainer_bench_r_build_cmd)"
    fi
    return 0
  fi
  if [[ -n "${SIF_R_BENCH:-}" ]]; then
    log "WARN: SIF_R_BENCH=${SIF_R_BENCH} missing"
  fi
  if [[ "${BENCH_R_FORCE_SIF:-0}" != "1" ]] && sand="$(apptainer_bench_resolve_r_sandbox)"; then
    export SIF_R_BENCH="${sand}"
    log "R bench image: ${sand} (sandbox; same as ${sif}, no --unsquash)"
    return 0
  fi
  export SIF_R_BENCH="${sif}"
  log "R bench image: ${sif} (sif; --unsquash)"
  if [[ ! -d "${sand}" ]]; then
    log "WARN: if unsquash fails (permission denied), build once: $(apptainer_bench_r_build_cmd)"
  fi
  return 0
}

apptainer_bench_ensure_r_export_image() {
  local sif sand
  sif="$(apptainer_bench_r_sif)"
  if [[ -n "${SIF_R_EXPORT:-}" && ( -f "${SIF_R_EXPORT}" || -d "${SIF_R_EXPORT}" ) ]]; then
    return 0
  fi
  if [[ -n "${SIF_R_EXPORT:-}" ]]; then
    log "WARN: SIF_R_EXPORT=${SIF_R_EXPORT} missing"
  fi
  if [[ "${BENCH_R_FORCE_SIF:-0}" != "1" ]] && sand="$(apptainer_bench_resolve_r_sandbox)"; then
    export SIF_R_EXPORT="${sand}"
    return 0
  fi
  export SIF_R_EXPORT="${sif}"
  return 0
}

apptainer_bench_bind=(
  --bind "${REPO}:${REPO}"
  --bind /iridisfs/ddnb:/iridisfs/ddnb
  --bind /scratch:/scratch
  --bind /iridisfs/ixsoftware:/iridisfs/ixsoftware
)

apptainer_bench_has_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

# scfast CUDA (NVRTC/cuBLAS) needs the CUDA toolkit on LD_LIBRARY_PATH, not just nvidia-smi.
# GPU Leiden in scfast-gpu needs libcugraph_c.so (RAPIDS); not in the CUDA toolkit module.
apptainer_bench_ensure_cugraph_lib() {
  local engines="${1:-}"
  [[ ",${engines}," == *",scfast-gpu,"* ]] || return 0
  if [[ -n "${SCFAST_CUGRAPH_LIB:-}" && -f "${SCFAST_CUGRAPH_LIB}" ]]; then
    return 0
  fi
  local gpu_img="${SIF_GPU_BENCH:-$(apptainer_bench_rsc_sandbox)}"
  local candidates=(
    "${gpu_img}/opt/conda/lib/libcugraph_c.so"
    "${gpu_img}/opt/conda/lib64/libcugraph_c.so"
  )
  local lib dir
  for lib in "${candidates[@]}"; do
    if [[ -f "${lib}" ]]; then
      export SCFAST_CUGRAPH_LIB="${lib}"
      dir="$(dirname "${lib}")"
      export LD_LIBRARY_PATH="${dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      log "scfast-gpu cuGraph: SCFAST_CUGRAPH_LIB=${SCFAST_CUGRAPH_LIB}"
      return 0
    fi
  done
  log "WARN: libcugraph_c.so not found under ${gpu_img}; scfast-gpu Leiden may fail"
  log "WARN: set SCFAST_CUGRAPH_LIB=/path/to/libcugraph_c.so (see docs/INSTALL.md)"
}

apptainer_bench_ensure_cuda_toolkit() {
  local engines="${1:-}"
  [[ ",${engines}," == *",scfast-gpu,"* ]] || return 0
  if [[ -n "${BENCH_CUDA_MODULE:-}" ]]; then
    module load "${BENCH_CUDA_MODULE}" 2>/dev/null || log "WARN: module load ${BENCH_CUDA_MODULE} failed"
    return 0
  fi
  local m
  for m in cuda/12.4.0 cuda/12.6.3 cuda/12.8.0 cuda/12.5.1 cuda/12.2.0; do
    if module load "${m}" 2>/dev/null; then
      log "CUDA toolkit for scfast-gpu: module load ${m}"
      return 0
    fi
  done
  log "WARN: could not module load cuda/*; scfast-gpu may fail (libnvrtc not found). Set BENCH_CUDA_MODULE=cuda/12.4.0"
}

apptainer_bench_ensure_gpu_image() {
  local d
  d="$(apptainer_bench_rsc_sandbox)"
  if [[ -z "${SIF_GPU_BENCH:-}" ]]; then
    [[ -d "${d}" ]] && export SIF_GPU_BENCH="${d}"
    return 0
  fi
  if [[ -f "${SIF_GPU_BENCH}" || -d "${SIF_GPU_BENCH}" ]]; then
    return 0
  fi
  log "WARN: SIF_GPU_BENCH=${SIF_GPU_BENCH} missing"
  if [[ -d "${d}" ]]; then
    export SIF_GPU_BENCH="${d}"
    log "WARN: using ${d}"
  else
    unset SIF_GPU_BENCH
  fi
}

apptainer_bench_default_install_rapids() {
  local engines="${1:-}"
  apptainer_bench_engines_need_rapids "${engines}" || return 0
  [[ -n "${BENCH_INSTALL_RAPIDS:-}" ]] && return 0
  apptainer_bench_ensure_gpu_image
  if [[ -n "${SIF_GPU_BENCH:-}" && ( -f "${SIF_GPU_BENCH}" || -d "${SIF_GPU_BENCH}" ) ]]; then
    export BENCH_INSTALL_RAPIDS=0
    log "BENCH_INSTALL_RAPIDS=0 (GPU python via Apptainer: ${SIF_GPU_BENCH})"
  else
    export BENCH_INSTALL_RAPIDS=1
    log "BENCH_INSTALL_RAPIDS=1 (pip into venv; or build $(apptainer_bench_rsc_sandbox))"
  fi
}

apptainer_bench_resolve_gpu_sif() {
  apptainer_bench_ensure_gpu_image
  local cand
  for cand in \
    "${SIF_GPU_BENCH:-}" \
    "$(apptainer_bench_rsc_sandbox)" \
    "/scratch/${USER}/rsc_cu12_sandbox" \
    "/scratch/${USER}/rsc_cu12.sif" \
    "/scratch/${USER}/rapids-singlecell-cu12.sif" \
    "/iridisfs/ddnb/images/rsc_cu12.sif" \
    "/scratch/${USER}/scpipe/scpipe.sif" \
    "/scratch/${USER}/scpipe.sif" \
    "/iridisfs/ddnb/images/scpipe.sif"; do
    [[ -n "${cand}" && ( -f "${cand}" || -d "${cand}" ) ]] && { printf '%s\n' "${cand}"; return 0; }
  done
  return 1
}

apptainer_bench_image_is_sandbox() {
  [[ -d "${1:-}" ]]
}

apptainer_bench_image_is_sif() {
  local img="${1:-}"
  [[ -f "${img}" && "${img}" == *.sif ]]
}

apptainer_bench_image_kind() {
  local img="${1:-}"
  if apptainer_bench_image_is_sandbox "${img}"; then
    printf '%s\n' sandbox
  elif apptainer_bench_image_is_sif "${img}"; then
    printf '%s\n' sif
  else
    printf '%s\n' unknown
  fi
}

# SIF: use APPTAINER_EXTRA (default --unsquash on Iridis). Sandbox dir: exec directly, no unsquash.
apptainer_bench_exec_flags() {
  local img="${1:-}"
  if apptainer_bench_image_is_sandbox "${img}"; then
    return 0
  fi
  if apptainer_bench_image_is_sif "${img}" || [[ -f "${img}" ]]; then
    printf '%s' "${APPT_EXTRA[@]}"
  fi
}

apptainer_bench_gpu_appt_flags() {
  local img="${1:-${SIF_GPU_BENCH:-}}"
  printf '%s' --nv
  local extra
  extra="$(apptainer_bench_exec_flags "${img}")"
  [[ -n "${extra}" ]] && printf ' %s' "${extra}"
}

# SIF --unsquash extracts under APPTAINER_TMPDIR (must be writable; /scratch/$USER usually works on Iridis X).
apptainer_bench_init_apptainer_tmp() {
  export APPTAINER_TMPDIR="${BENCH_APPTAINER_TMPDIR:-/scratch/${USER}/apptainer-tmp}"
  mkdir -p "${APPTAINER_TMPDIR}"
}

apptainer_bench_default_engines() {
  if apptainer_bench_has_gpu; then
    printf '%s\n' "seurat,scanpy-gpu,scfast-gpu"
  else
    printf '%s\n' "seurat,scanpy,scfast-cpu"
  fi
}

apptainer_bench_engines_need_scfast() {
  [[ ",${1}," == *",scfast-cpu,"* || ",${1}," == *",scfast-gpu,"* ]]
}

apptainer_bench_engines_need_rapids() {
  [[ ",${1}," == *",scanpy-gpu,"* || ",${1}," == *",rapids,"* ]]
}

apptainer_bench_scfast_maturin_features() {
  local engines="${1:-}"
  if [[ ",${engines}," == *",scfast-gpu,"* ]] && apptainer_bench_has_gpu; then
    printf '%s' "cuda,cugraph"
  fi
}

apptainer_bench_scfast_import_ok() {
  local engines="${1:-}"
  local py="${BENCH_VENV}/bin/python3"
  "${py}" -c "import scfast" 2>/dev/null || return 1
  if [[ ",${engines}," == *",scfast-gpu,"* ]]; then
    "${py}" -c 'import scfast; i=scfast.device_info(); raise SystemExit(0 if i.get("cuda_available") else 1)' 2>/dev/null || return 1
  fi
  return 0
}

# Prebuilt wheels: https://rapids-singlecell.scverse.org/en/latest/installation.html
apptainer_bench_rapids_pip_spec() {
  local cuda_major="12"
  if command -v nvidia-smi >/dev/null 2>&1; then
    local drv
    drv="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\).*/\1/p' | head -1)"
    [[ -n "${drv}" ]] && cuda_major="${drv}"
  fi
  if [[ "${BENCH_RAPIDS_CUDA:-}" == "13" ]] || [[ "${cuda_major}" -ge 13 ]]; then
    printf '%s\n' "rapids-singlecell-cu13[rapids]"
  else
    printf '%s\n' "rapids-singlecell-cu12[rapids]"
  fi
}

apptainer_bench_pip_install_rapids() {
  local py="${BENCH_VENV}/bin/python3"
  local spec logf
  spec="$(apptainer_bench_rapids_pip_spec)"
  logf="${BENCH_ROOT}/logs/bootstrap_rapids.log"
  mkdir -p "${BENCH_ROOT}/logs"
  log "pip install ${spec} (NVIDIA index; see rapids-singlecell installation docs)"
  "${py}" -m pip install -U pip wheel 2>&1 | tee -a "${logf}" || true
  "${py}" -m pip install "${spec}" --extra-index-url=https://pypi.nvidia.com \
    2>&1 | tee -a "${logf}"
}

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

# maturin develop: needs VIRTUAL_ENV; conda (base) must be unset or it errors.
bench_maturin_develop() {
  local crate_dir="${1:?}"
  local features="${2:-}"
  local majmin
  majmin="$("${BENCH_VENV}/bin/python3" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ ! -e "${BENCH_VENV}/bin/python${majmin}" ]]; then
    ln -sfn python3 "${BENCH_VENV}/bin/python${majmin}"
  fi
  (
    unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_SHLVL
    export VIRTUAL_ENV="${BENCH_VENV}"
    export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/scratch/${USER}/cargo-target}"
    export PATH="${BENCH_VENV}/bin:${CARGO_HOME:-/scratch/${USER}/cargo}/bin:${PATH}"
    cd "${crate_dir}"
    if [[ -n "${features}" ]]; then
      "${BENCH_VENV}/bin/python3" -m maturin develop --release --features "${features}"
    else
      "${BENCH_VENV}/bin/python3" -m maturin develop --release
    fi
  )
}

# scfast Rust engine (bench/run_py.py --engine scfast-cpu); optional if rust/maturin missing.
apptainer_bench_bootstrap_scfast() {
  local stamp="${BENCH_VENV}/.scfast_bench_ok"
  local engines="${ENGINES:-}"
  if [[ "${BENCH_SCFAST_FORCE_REBUILD:-0}" != "1" ]] && [[ -f "${stamp}" ]] && apptainer_bench_scfast_import_ok "${engines}"; then
    log "scfast already built (${stamp})"
    return 0
  fi
  if [[ "${BENCH_SCFAST_FORCE_REBUILD:-0}" == "1" ]]; then
    log "BENCH_SCFAST_FORCE_REBUILD=1; re-running maturin"
    rm -f "${stamp}"
  fi
  if [[ "${BENCH_SCFAST_FORCE_REBUILD:-0}" != "1" ]] && apptainer_bench_scfast_import_ok "${engines}"; then
    touch "${stamp}"
    log "scfast already importable in ${BENCH_VENV}"
    return 0
  fi
  [[ "${BENCH_BUILD_SCFAST:-1}" == "1" ]] || { log "scfast not installed (BENCH_BUILD_SCFAST=0)"; return 0; }
  log "one-time scfast build into ${BENCH_VENV} (gcc+rustc; will not re-run after ${stamp})"
  mkdir -p "${BENCH_ROOT}/logs" "${CARGO_TARGET_DIR:-/scratch/${USER}/cargo-target}"
  module load gcc/13.3.0 2>/dev/null || module load gcc 2>/dev/null || true
  export RUSTUP_HOME="${RUSTUP_HOME:-/scratch/${USER}/rustup}"
  export CARGO_HOME="${CARGO_HOME:-/scratch/${USER}/cargo}"
  export PATH="${CARGO_HOME}/bin:${PATH}"
  # shellcheck source=bench/rustup_install.sh
  source "${REPO}/bench/rustup_install.sh"
  if ! bench_cargo_ok && [[ "${BENCH_INSTALL_RUST:-0}" == "1" ]]; then
    log "installing rustup on scratch (one-time; BENCH_INSTALL_RUST=1)"
    bench_ensure_rustup
  fi
  if ! bench_cargo_ok; then
    log "WARN: no working cargo; run once: ./bench/rustup_install.sh && ./bench/install_scfast.sh"
    return 0
  fi
  if ! "${BENCH_VENV}/bin/python3" -m pip install -q maturin 2>&1 | tee -a "${BENCH_ROOT}/logs/bootstrap_scfast.log"; then
    log "WARN: maturin install failed"
    return 0
  fi
  local feats=""
  feats="$(apptainer_bench_scfast_maturin_features "${engines}")"
  [[ -n "${feats}" ]] && log "scfast maturin --features ${feats}"
  bench_maturin_develop "${REPO}/rust/scfast-py" "${feats}" 2>&1 | tee -a "${BENCH_ROOT}/logs/bootstrap_scfast.log" || {
    log "WARN: scfast build failed; see ${BENCH_ROOT}/logs/bootstrap_scfast.log"
    return 0
  }
  "${BENCH_VENV}/bin/python3" -c "import scfast; print('scfast', scfast.__version__)"
  touch "${stamp}"
}

# rapids-singlecell for scanpy-gpu / rapids (host venv or GPU apptainer image + --nv).
apptainer_bench_bootstrap_rapids() {
  local py="${BENCH_VENV}/bin/python3"
  local rstamp="${BENCH_VENV}/.rapids_bench_ok"
  BENCH_RAPIDS_PYTHON="${py}"
  if [[ -f "${rstamp}" ]] && "${py}" -c "import rapids_singlecell, cupy" 2>/dev/null; then
    log "rapids-singlecell ready (${rstamp})"
    return 0
  fi
  if "${py}" -c "import rapids_singlecell, cupy" 2>/dev/null; then
    touch "${rstamp}"
    log "rapids-singlecell in ${BENCH_VENV}"
    return 0
  fi
  apptainer_bench_ensure_gpu_image
  local sif=""
  sif="$(apptainer_bench_resolve_gpu_sif || true)"
  if [[ -n "${sif}" ]]; then
    # Sandbox/SIF: exec with --nv + conda base (avoid --no-home; CuPy/numba need writable $HOME).
    # shellcheck disable=SC2046
    if "${APPT_BIN}" exec $(apptainer_bench_gpu_appt_flags "${sif}") \
        "${apptainer_bench_bind[@]}" "${sif}" \
        bash -c 'export PYTHONNOUSERSITE=1 NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/scratch/${USER}/bench/numba-cache}"; mkdir -p "$NUMBA_CACHE_DIR"; source /opt/conda/etc/profile.d/conda.sh && conda activate base && exec python "$@"' _ \
        -c "import rapids_singlecell, cupy" 2>/dev/null; then
      export SIF_GPU_BENCH="${sif}"
      log "GPU bench image: ${sif} ($(apptainer_bench_image_kind "${sif}"); rapids via apptainer --nv)"
      return 0
    fi
    log "WARN: ${sif} missing rapids_singlecell (or GPU/CuPy probe failed; need nvidia-smi on this node)"
    unset SIF_GPU_BENCH
  fi
  if [[ "${BENCH_INSTALL_RAPIDS:-0}" == "1" ]] && apptainer_bench_has_gpu; then
    apptainer_bench_pip_install_rapids || true
    if "${py}" -c "import rapids_singlecell, cupy" 2>/dev/null; then
      log "rapids-singlecell installed in ${BENCH_VENV}"
      touch "${BENCH_VENV}/.rapids_bench_ok"
      return 0
    fi
    log "WARN: pip rapids install failed; see ${BENCH_ROOT}/logs/bootstrap_rapids.log"
  fi
  if apptainer_bench_has_gpu; then
    log "WARN: GPU visible but no rapids env"
    log "WARN:   BENCH_INSTALL_RAPIDS=1 ./run_pbmc3k_apptainer.sh"
    log "WARN:   apptainer build --sandbox $(apptainer_bench_rsc_sandbox) docker://ghcr.io/scverse/rapids-singlecell-cu12:latest"
  fi
}

# Exit 1 if ENGINES lists GPU python engines but no RAPIDS_PYTHON wrapper / venv rapids.
apptainer_bench_require_gpu_python() {
  local engines="${1:-}"
  [[ "${engines}" == *"scanpy-gpu"* || "${engines}" == *"rapids"* ]] || return 0
  local probe="${BENCH_RAPIDS_PYTHON:-}"
  [[ -n "${probe}" && -x "${probe}" ]] || probe="${WRAP}/python3-gpu"
  if [[ -x "${probe}" ]]; then
    if "${probe}" -c "import rapids_singlecell, cupy" 2>/dev/null; then
      return 0
    fi
  fi
  if "${BENCH_VENV}/bin/python3" -c "import rapids_singlecell, cupy" 2>/dev/null; then
    return 0
  fi
  log "ERROR: ENGINES need rapids-singlecell (scanpy-gpu / rapids in: ${engines}) but none is installed"
  log "ERROR: run on a GPU node (nvidia-smi -L), image $(apptainer_bench_rsc_sandbox), or ENGINES=seurat only"
  return 1
}

# Exit 1 if ENGINES lists scfast-gpu but the venv scfast lacks a usable CUDA backend.
apptainer_bench_require_scfast_gpu() {
  local engines="${1:-}"
  [[ "${engines}" == *"scfast-gpu"* ]] || return 0
  if apptainer_bench_scfast_import_ok "${engines}"; then
    return 0
  fi
  local err=""
  err="$("${BENCH_VENV}/bin/python3" -c 'import scfast, json; print(scfast.device_info().get("cuda_error") or "")' 2>/dev/null || true)"
  log "ERROR: ENGINES include scfast-gpu but scfast CUDA is not usable (${err:-unknown})"
  log "ERROR: nvidia-smi is not enough; try: module load cuda/12.4.0  (or export BENCH_CUDA_MODULE=cuda/12.4.0)"
  log "ERROR: then: /scratch/\$USER/bench/venv/bin/python3 -c \"import scfast; print(scfast.device_info())\""
  if [[ ",${engines}," == *",scfast-gpu,"* ]]; then
    log "ERROR: GPU Leiden also needs libcugraph_c.so (export SCFAST_CUGRAPH_LIB or SIF_GPU_BENCH rapids sandbox)"
  fi
  return 1
}

# leidenbase for Seurat >= 5.1 (avoids reticulate when available).
apptainer_bench_bootstrap_leidenbase() {
  local stamp="${BENCH_ROOT}/.leidenbase_ok"
  [[ -f "$stamp" ]] && return 0
  log "bootstrap R leidenbase (image ${SIF_R_BENCH})"
  mkdir -p "${BENCH_ROOT}/logs"
  # shellcheck disable=SC2046
  "${APPT_BIN}" exec $(apptainer_bench_exec_flags "${SIF_R_BENCH}") "${apptainer_bench_bind[@]}" "${SIF_R_BENCH}" \
    /usr/local/bin/Rscript -e '
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
  local _r_flags _gpu_flags
  _r_flags="$(apptainer_bench_exec_flags "${SIF_R_BENCH}")"
  # printf only (heredocs + set -u break on "$@" with no positional params).
  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' '# Clustering uses leidenbase in bench/run_seurat.R (no reticulate/leidenalg in SIF).'
    printf 'exec %s exec %s \\\n' "${APPT_BIN}" "${_r_flags}"
    printf '  --bind %s:%s --bind /iridisfs/ddnb:/iridisfs/ddnb --bind /scratch:/scratch \\\n' "${REPO}" "${REPO}"
    printf '  --bind /iridisfs/ixsoftware:/iridisfs/ixsoftware \\\n'
    printf '  %s \\\n' "${SIF_R_BENCH}"
    printf '%s\n' '  /usr/bin/env -u VIRTUAL_ENV -u RETICULATE_PYTHON -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \'
    printf '%s\n' '  /usr/local/bin/Rscript "$@"'
  } > "${WRAP}/Rscript"
  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf 'exec "%s/bin/python3" "$@"\n' "${BENCH_VENV}"
  } > "${WRAP}/python3"
  chmod +x "${WRAP}/Rscript" "${WRAP}/python3"
  if [[ -n "${SIF_GPU_BENCH:-}" && ( -f "${SIF_GPU_BENCH}" || -d "${SIF_GPU_BENCH}" ) ]]; then
    _gpu_flags="$(apptainer_bench_gpu_appt_flags "${SIF_GPU_BENCH}")" || _gpu_flags="--nv"
    {
      printf '%s\n' '#!/usr/bin/env bash'
      printf 'exec %s exec %s \\\n' "${APPT_BIN}" "${_gpu_flags}"
      printf '  --bind %s:%s --bind /iridisfs/ddnb:/iridisfs/ddnb --bind /scratch:/scratch \\\n' "${REPO}" "${REPO}"
      printf '  --bind /iridisfs/ixsoftware:/iridisfs/ixsoftware \\\n'
      printf '  %s \\\n' "${SIF_GPU_BENCH}"
      printf '%s\n' '  bash -c '"'"'export PYTHONNOUSERSITE=1 NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/scratch/${USER}/bench/numba-cache}"; mkdir -p "$NUMBA_CACHE_DIR"; source /opt/conda/etc/profile.d/conda.sh && conda activate base && exec python "$@"'"'"' _ "$@"'
    } > "${WRAP}/python3-gpu"
    chmod +x "${WRAP}/python3-gpu"
    export BENCH_RAPIDS_PYTHON="${WRAP}/python3-gpu"
    log "RAPIDS_PYTHON -> ${BENCH_RAPIDS_PYTHON}"
  else
    export BENCH_RAPIDS_PYTHON="${BENCH_VENV}/bin/python3"
  fi
  log "wrappers written under ${WRAP}"
}
