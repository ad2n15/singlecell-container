#!/bin/bash
# Update an unpacked scpipe image (Apptainer sandbox) in place from this checkout: scpipe's code
# under /opt/scpipe, the scpipe Python package, and scfast + scfastR rebuilt from rust/.
# Minutes instead of the ~1 h full build; rebuild the image (scpipe-build.def) instead when the
# base image, R/Python dependency lists or system libraries change.
#
#   containers/update_sandbox.sh /iridisfs/ddnb/images/scpipe_sandbox             # code + Rust
#   containers/update_sandbox.sh /iridisfs/ddnb/images/scpipe_sandbox --code-only # no Rust rebuild
#
# Make the sandbox from a built image on a local disk, then copy it with rsync -a: on GPFS
# (Iridis /iridisfs) files created by extraction inherit an ACL without execute permission.
#   apptainer build --sandbox /tmp/$USER-sbx scpipe.sif && rsync -a /tmp/$USER-sbx/ <sandbox>/
#
# Needs internet (Rust toolchain and crates), so run it on a login node. Build work happens in
# /tmp inside the read-only sandbox; results are copied in from outside with rsync -a, which
# sets each file's mode explicitly.
set -euo pipefail

usage() { echo "usage: $0 <sandbox dir> [--code-only]" >&2; exit 2; }
[ $# -ge 1 ] || usage
SBX=$(realpath "$1"); CODE_ONLY=0
[ "${2:-}" = "--code-only" ] && CODE_ONLY=1
[ -d "$SBX/opt/scpipe" ] || { echo "$SBX is not an scpipe sandbox (no /opt/scpipe)" >&2; exit 1; }
REPO=$(cd "$(dirname "$0")/.." && pwd)
command -v apptainer >/dev/null || module load apptainer 2>/dev/null || true

SP=$(ls -d "$SBX"/opt/conda/lib/python3.*/site-packages | head -1)
RLIB=$SBX/usr/local/lib/R/site-library
W=$(mktemp -d "/tmp/$USER-scpipe-update.XXXXXX")
trap 'rm -rf "$W"' EXIT

echo "== source $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')$(git -C "$REPO" diff --quiet 2>/dev/null || echo ' (with uncommitted changes)')"
rsync -a --exclude target --exclude __pycache__ --exclude '*.o' --exclude '*.so' \
  "$REPO/python" "$REPO/rust" "$W/"

echo "== build inside the sandbox"
apptainer exec --cleanenv -B "$W:/work" --env CODE_ONLY=$CODE_ONLY "$SBX" bash -c '
set -euo pipefail
cd /work
SYS_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
if [ "$CODE_ONLY" = 0 ]; then
  export RUSTUP_HOME=/work/rustup CARGO_HOME=/work/cargo CARGO_TARGET_DIR=/work/cargo-target
  # --no-modify-path: $HOME is the real home inside the sandbox; leave its shell profiles alone.
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --quiet --no-modify-path --profile minimal --default-toolchain stable
  /opt/conda/bin/pip install --quiet --no-cache-dir --target /work/tools "maturin>=1.9.4,<2"
  PATH=/work/cargo/bin:/work/tools/bin:/opt/conda/bin:$PATH PYTHONPATH=/work/tools \
    maturin build --release --locked -m rust/scfast-py/Cargo.toml --features cuda,cugraph -o /work/wheels
  mkdir -p /work/rlib
  PATH=/work/cargo/bin:$SYS_PATH SCFASTR_CARGO_FEATURES=cuda,cugraph SCFASTR_TARGET_DIR=/work/scfastr-target \
    R CMD INSTALL --no-test-load -l /work/rlib rust/scfast-r
fi
/opt/conda/bin/pip install --quiet --no-cache-dir --no-deps --target /work/py /work/python $(ls /work/wheels/scfast-*.whl 2>/dev/null)
'

echo "== install into $SBX"
for d in python R conf config docs tests containers; do
  rsync -a --delete --exclude __pycache__ "$REPO/$d/" "$SBX/opt/scpipe/$d/"
done
pkgs=(scpipe); [ $CODE_ONLY = 0 ] && pkgs+=(scfast)
for p in "${pkgs[@]}"; do
  rm -rf "$SP/$p" "$SP"/"$p"-*.dist-info
  rsync -a "$W/py/$p" "$W"/py/"$p"-*.dist-info "$SP/"
done
[ $CODE_ONLY = 0 ] && rsync -a --delete "$W/rlib/scfastR/" "$RLIB/scfastR/"
chmod -R a+rX "$SBX/opt/scpipe"
{ echo "updated $(date -u +%FT%TZ) from $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  git -C "$REPO" diff --quiet 2>/dev/null || echo "with uncommitted changes"; } > "$SBX/opt/scpipe/UPDATED"

echo "== check"
apptainer exec --cleanenv "$SBX" bash -c '
scpipe --help >/dev/null
python -c "import scpipe, scfast; print(\"scfast\", scfast.__file__)"
Rscript -e "stopifnot(requireNamespace(\"scfastR\", quietly = TRUE)); cat(\"scfastR\", format(packageVersion(\"scfastR\")), \"\n\")"'
cat "$SBX/opt/scpipe/UPDATED"
