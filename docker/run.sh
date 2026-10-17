#!/usr/bin/env bash
# Build the image and run a mode in it. Results from `bench` appear in bench/results/ on the host.
#   bash docker/run.sh test                       # whole test suite
#   STAGES="rust pytest" bash docker/run.sh test  # only some stages
#   CELLS=100000 THREADS=8 MEM=48g bash docker/run.sh bench
#   bash docker/run.sh all                        # tests, then the benchmark
#   bash docker/run.sh shell
# MEM and CPUS cap the container (Seurat needs tens of GB at 100k cells).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-test}"
docker build -f docker/Dockerfile -t scfast .
mkdir -p bench/results
tty=(); [[ -t 0 ]] && tty=(-it)
docker run --rm "${tty[@]}" ${MEM:+--memory "$MEM"} ${CPUS:+--cpus "$CPUS"} \
  -e CELLS -e GENES -e TYPES -e THREADS -e REPS -e ENGINES -e NAME -e TIMEOUT -e SKIP_UMAP -e STAGES -e SKIP \
  -v "$PWD/bench/results:/work/bench/results" scfast "$mode"
