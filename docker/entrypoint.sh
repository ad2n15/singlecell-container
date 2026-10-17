#!/usr/bin/env bash
# Container entry point: test | bench | all | shell
#   test   tests/run_all.sh (every stage; STAGES="rust pytest" to pick some, SKIP="bench-smoke" to omit)
#   bench  synthetic benchmark: CELLS=100000 GENES=5000 TYPES=10 THREADS=$(nproc) REPS=3
#          ENGINES=seurat,scanpy,scfast-cpu NAME=run_$CELLS SKIP_UMAP=0 TIMEOUT=7200
#   all    test, then bench
#   shell  interactive bash
set -uo pipefail
cd /work
mode="${1:-test}"

run_tests() { bash tests/run_all.sh ${STAGES:-}; }

run_bench() {
  CELLS="${CELLS:-100000}" GENES="${GENES:-5000}" TYPES="${TYPES:-10}" THREADS="${THREADS:-$(nproc)}"
  REPS="${REPS:-3}" ENGINES="${ENGINES:-seurat,scanpy,scfast-cpu}" NAME="${NAME:-run_${CELLS}}" TIMEOUT="${TIMEOUT:-7200}"
  echo "== environment"; nproc; free -g | sed -n 2p
  python -c "import scfast, scanpy; print('scfast ok, scanpy', scanpy.__version__)"
  Rscript -e 'cat("Seurat", as.character(packageVersion("Seurat")), "\n")'
  mkdir -p data
  [[ -f data/synth_$CELLS.h5 ]] || python bench/synth.py --cells "$CELLS" --genes "$GENES" --types "$TYPES" --out data/synth_$CELLS.h5
  extra=(); [[ "${SKIP_UMAP:-0}" == 1 ]] && extra+=(--skip-umap)
  bash bench/run_bench.sh --input data/synth_$CELLS.h5 --labels data/synth_$CELLS.h5.labels.tsv \
    --name "$NAME" --sizes "$CELLS" --engines "$ENGINES" --reps "$REPS" --threads "$THREADS" \
    --timeout "$TIMEOUT" --qc "--max-mt 100 --max-genes 1e9" "${extra[@]}"
  echo "== done: bench/results/$NAME/report.md"
}

case "$mode" in
  test)  run_tests ;;
  bench) run_bench ;;
  all)   run_tests; rc=$?; run_bench; exit $(( rc || $? )) ;;
  shell) exec bash ;;
  *) echo "usage: test | bench | all | shell" >&2; exit 2 ;;
esac
