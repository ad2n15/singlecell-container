#!/usr/bin/env bash
# Build the image and run the benchmark in a container, leaving results in bench/results/.
#   bash bench/docker/run.sh                      # 100k cells, 3 reps, all CPUs
#   CELLS=100000 THREADS=8 MEM=48g SKIP_UMAP=1 bash bench/docker/run.sh
# Give the container real resources: Seurat needs tens of GB at 100k.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
docker build -f bench/docker/Dockerfile -t scfast-bench .
mkdir -p bench/results
docker run --rm ${MEM:+--memory "$MEM"} ${CPUS:+--cpus "$CPUS"} \
  -e CELLS -e GENES -e TYPES -e THREADS -e REPS -e ENGINES -e NAME -e TIMEOUT -e SKIP_UMAP \
  -v "$PWD/bench/results:/work/bench/results" scfast-bench
