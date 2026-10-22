#!/usr/bin/env bash
# Download and run rustup installer (Iridis curl lacks --proto). Source or execute.
set -euo pipefail

export RUSTUP_HOME="${RUSTUP_HOME:-/scratch/${USER}/rustup}"
export CARGO_HOME="${CARGO_HOME:-/scratch/${USER}/cargo}"
mkdir -p "${RUSTUP_HOME}" "${CARGO_HOME}"

bench_rustup_fetch() {
  local url="https://sh.rustup.rs"
  if curl -fsSL "$url" 2>/dev/null; then
    return 0
  fi
  if wget -qO- "$url" 2>/dev/null; then
    return 0
  fi
  echo "ERROR: need curl or wget to fetch ${url}" >&2
  return 1
}

bench_cargo_ok() {
  command -v cargo >/dev/null 2>&1 && cargo --version >/dev/null 2>&1
}

bench_ensure_rustup() {
  export PATH="${CARGO_HOME}/bin:${PATH}"
  if [[ -f "${CARGO_HOME}/env" ]]; then
    # shellcheck disable=SC1091
    source "${CARGO_HOME}/env"
  fi
  if bench_cargo_ok; then
    return 0
  fi
  echo "Installing rustup (stable) into ${CARGO_HOME} (RUSTUP_HOME=${RUSTUP_HOME}) ..."
  bench_rustup_fetch | env RUSTUP_HOME="${RUSTUP_HOME}" CARGO_HOME="${CARGO_HOME}" \
    sh -s -- -y --default-toolchain stable
  export PATH="${CARGO_HOME}/bin:${PATH}"
  if ! bench_cargo_ok; then
    echo "ERROR: rustup install failed (check quota on ${RUSTUP_HOME} and ${CARGO_HOME})" >&2
    return 1
  fi
  cargo --version
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  bench_ensure_rustup
fi
