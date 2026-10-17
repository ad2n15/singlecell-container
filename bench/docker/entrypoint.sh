#!/usr/bin/env bash
# Generate a synthetic dataset and benchmark it. Configure with environment variables:
#   CELLS=100000 GENES=5000 TYPES=10 THREADS=8 REPS=3 ENGINES=seurat,scanpy,scfast-cpu NAME=run_100k
#   SKIP_UMAP=1 to leave UMAP out of the timings; UMAP_ONLY is not supported.
set -euo pipefail
CELLS="${CELLS:-100000}" GENES="${GENES:-5000}" TYPES="${TYPES:-10}" THREADS="${THREADS:-$(nproc)}"
REPS="${REPS:-3}" ENGINES="${ENGINES:-seurat,scanpy,scfast-cpu}" NAME="${NAME:-run_${CELLS}}"
TIMEOUT="${TIMEOUT:-7200}"
OUT=/work/bench/results/$NAME
mkdir -p /work/data "$OUT"
echo "== environment"; nproc; free -g | sed -n 2p
python -c "import scfast, scanpy; print('scfast ok, scanpy', scanpy.__version__)"
Rscript -e 'cat("Seurat", as.character(packageVersion("Seurat")), "\n")'
if [[ ! -f /work/data/synth_$CELLS.h5 ]]; then
  python bench/synth.py --cells "$CELLS" --genes "$GENES" --types "$TYPES" --out /work/data/synth_$CELLS.h5
fi
EXTRA=()
[[ "${SKIP_UMAP:-0}" == 1 ]] && EXTRA+=(--skip-umap)
bash bench/run_bench.sh --input /work/data/synth_$CELLS.h5 --labels /work/data/synth_$CELLS.h5.labels.tsv \
  --name "$NAME" --sizes "$CELLS" --engines "$ENGINES" --reps "$REPS" --threads "$THREADS" \
  --timeout "$TIMEOUT" --qc "--max-mt 100 --max-genes 1e9" "${EXTRA[@]}"
echo "== done: $OUT/report.md"
