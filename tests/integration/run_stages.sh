#!/usr/bin/env bash
# End-to-end test for stages S0-S2 (docs/stages-spec.md section 6).
#
#   tests/integration/run_stages.sh [WORKDIR]
#
# Runs `scpipe run` through S2 on the two-sample QC-stress fixture
# (tests/fixtures/stage_inputs: donor_A as a 10x mtx folder, donor_B as .h5ad)
# twice, once with device: cpu and once with device: gpu. Without a GPU the
# gpu run uses gpu_core's CPU fallback through the test-only override
# SCPIPE_ALLOW_CPU_AS_GPU=1. Then checks the outputs, reruns, and a
# resolution change.
#
# Every step runs even if an earlier one fails (steps that depend on a failed
# one are marked SKIP). Prints a PASS/FAIL table with timings and exits
# non-zero if anything failed or was skipped. Without WORKDIR a temp dir is
# used and removed on success (kept on failure, or always with KEEP_WORKDIR=1).
#
# Environment: PYTHON, RSCRIPT (interpreters); MIN_LOWQ_EXCLUDED,
# MIN_DOUBLET_EXCLUDED, MIN_PLATELET_REVIEW, MIN_NONE_KEPT, MIN_PURITY, MIN_ARI
# (thresholds, see check_stages.py).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export RSCRIPT="${RSCRIPT:-Rscript}"
CONF="$HERE/stage_configs"
INPUTS="$REPO/tests/fixtures/stage_inputs"
CHECK=("$PYTHON" "$HERE/check_stages.py")
NEW_RESOLUTION=1.0

if [[ $# -ge 1 ]]; then
  WORK="$1"; mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/scpipe-stages.XXXXXX")"
  CLEANUP_ON_PASS=1
fi
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$WORK/.numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$WORK/.mpl}"
LOG="$WORK/logs"; mkdir -p "$LOG"
CPU="$WORK/run_cpu"; GPU="$WORK/run_gpu"

echo "scpipe stages S0-S2 end-to-end"
echo "  repo:    $REPO"
echo "  workdir: $WORK"
echo

declare -a NAMES STATUS TIMES
declare -A RESULT
T_ALL=$(date +%s.%N)
run_step() {   # run_step ID "description" DEPENDS_ON(space-separated or "") -- cmd...
  local id="$1" desc="$2" deps="$3"; shift 4
  local d
  for d in $deps; do
    if [[ "${RESULT[$d]:-}" != "PASS" ]]; then
      NAMES+=("$id  $desc"); STATUS+=("SKIP"); TIMES+=("-"); RESULT[$id]="SKIP"
      echo "=== [$id] $desc: SKIP (needs $d)"; echo; return
    fi
  done
  echo "=== [$id] $desc"
  local t0 t1 rc
  t0=$(date +%s.%N)
  "$@" 2>&1 | tee "$LOG/$id.log"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s.%N)
  local st="PASS"; [[ $rc -eq 0 ]] || st="FAIL"
  NAMES+=("$id  $desc"); STATUS+=("$st"); TIMES+=("$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1fs", b-a}')")
  RESULT[$id]="$st"
  echo "--- [$id] $st (exit $rc)"; echo
}

scpipe() { "$PYTHON" -m scpipe "$@"; }

# A pipeline run. On failure, show the tail of every stage log that exists.
pipeline_run() {   # pipeline_run RUN_DIR CONFIG [extra env assignments...]
  local run="$1" cfg="$2"; shift 2
  env "$@" "$PYTHON" -m scpipe run --config "$cfg" --run-dir "$run" --until S2
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "scpipe run exited $rc"
    local f
    for f in "$run"/logs/*.log; do
      [[ -e "$f" ]] || continue
      echo "----- tail of $f"; tail -n 40 "$f"
    done
    [[ -f "$run/state.json" ]] && { echo "----- state.json"; cat "$run/state.json"; }
  fi
  return $rc
}

status_check() {   # scpipe status prints every stage and exits 0
  local out rc s miss=0
  out="$(scpipe status --run-dir "$CPU" 2>&1)"; rc=$?
  echo "$out"
  [[ $rc -eq 0 ]] || { echo "scpipe status exited $rc"; return 1; }
  for s in S0 S1a S1b S1c S2; do
    grep -q "$s" <<<"$out" || { echo "status output does not mention $s"; miss=1; }
  done
  grep -qi "done" <<<"$out" || { echo "status output does not say done"; miss=1; }
  return $miss
}

rerun_same() {   # same config again: every stage must be skipped
  "${CHECK[@]}" snapshot "$CPU" "$WORK/state_before_rerun.json" || return 1
  pipeline_run "$CPU" "$CONF/cpu.yaml" || return 1
  "${CHECK[@]}" skipped "$WORK/state_before_rerun.json" "$CPU"
}

resume_noop() {  # resume on a finished run: exits 0, reruns nothing
  "${CHECK[@]}" snapshot "$CPU" "$WORK/state_before_resume.json" || return 1
  scpipe resume --run-dir "$CPU" || { echo "scpipe resume exited $?"; return 1; }
  "${CHECK[@]}" skipped "$WORK/state_before_resume.json" "$CPU"
}

rerun_resolution() {   # clustering.resolution changed: only S2 reruns
  # The copy lives in $WORK, so make its relative paths absolute first.
  local cfg="$WORK/configs/cpu_res.yaml"
  mkdir -p "$WORK/configs"
  sed -e "s#^samples: .*#samples: $INPUTS/samples.csv#" \
      -e "s#^qc_policy: .*#qc_policy: $CONF/qc_policy.yaml#" \
      -e "s#^  resolution: .*#  resolution: $NEW_RESOLUTION#" "$CONF/cpu.yaml" > "$cfg"
  diff "$CONF/cpu.yaml" "$cfg"
  grep -q "resolution: $NEW_RESOLUTION" "$cfg" || { echo "could not edit resolution"; return 1; }
  "${CHECK[@]}" snapshot "$CPU" "$WORK/state_before_resolution.json" || return 1
  pipeline_run "$CPU" "$cfg" || return 1
  "${CHECK[@]}" reran "$WORK/state_before_resolution.json" "$CPU" --stages S2 || return 1
  local r
  r="$("$PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s.get("resolution", s.get("clustering", {}).get("resolution") if isinstance(s.get("clustering"), dict) else None))' "$CPU/S2/summary.json")"
  echo "S2/summary.json resolution: $r"
  [[ "$r" == "None" || "$r" == "$NEW_RESOLUTION" ]] || { echo "summary resolution $r != $NEW_RESOLUTION"; return 1; }
}

run_step T0  "stage inputs reproducible (--check)"   ""      -- "$PYTHON" "$REPO/tests/fixtures/make_stage_inputs.py" --check
run_step T1  "scpipe doctor"                         ""      -- scpipe doctor --config "$CONF/cpu.yaml"
run_step C1  "run S0-S2, device cpu"                 "T0"    -- pipeline_run "$CPU" "$CONF/cpu.yaml"
run_step G1  "run S0-S2, device gpu (CPU fallback)"  "T0"    -- pipeline_run "$GPU" "$CONF/gpu.yaml" SCPIPE_ALLOW_CPU_AS_GPU=1
run_step C2  "cpu: state.json all done, logs, config copies" "C1" -- "${CHECK[@]}" state "$CPU"
run_step G2  "gpu: state.json all done, logs, config copies" "G1" -- "${CHECK[@]}" state "$GPU"
run_step C3  "cpu: S0 ingest report (two formats)"   "C1"    -- "${CHECK[@]}" ingest "$CPU" "$INPUTS"
run_step C4  "cpu: S1 metrics vs independent calc"   "C1"    -- "${CHECK[@]}" metrics "$CPU" "$INPUTS"
run_step C5  "cpu: QC decisions vs injected truth"   "C1"    -- "${CHECK[@]}" qc-truth "$CPU" "$INPUTS/truth.tsv"
run_step G5  "gpu: QC decisions vs injected truth"   "G1"    -- "${CHECK[@]}" qc-truth "$GPU" "$INPUTS/truth.tsv"
run_step X1  "QC decisions identical cpu vs gpu"     "C1 G1" -- "${CHECK[@]}" qc-same "$CPU" "$GPU"
run_step C6  "cpu: S2 object, summary, purity"       "C1"    -- "${CHECK[@]}" s2 "$CPU" "$INPUTS/truth.tsv" "$WORK/clusters_cpu.tsv" --device cpu
run_step G6  "gpu: S2 object, summary, purity"       "G1"    -- "${CHECK[@]}" s2 "$GPU" "$INPUTS/truth.tsv" "$WORK/clusters_gpu.tsv" --device gpu
run_step X2  "ARI cpu vs gpu clusters"               "C6 G6" -- "${CHECK[@]}" ari "$WORK/clusters_cpu.tsv" "$WORK/clusters_gpu.tsv"
run_step R0  "scpipe status"                         "C1"    -- status_check
run_step R1  "rerun same config skips every stage"   "C2"    -- rerun_same
run_step R2  "resume on a finished run is a no-op"   "R1"    -- resume_noop
run_step R3  "resolution change reruns only S2"      "R1"    -- rerun_resolution

T_END=$(date +%s.%N)
echo "================= stages S0-S2 summary ================="
printf "%-6s %-8s %s\n" "RESULT" "TIME" "STEP"
fail=0
for i in "${!NAMES[@]}"; do
  printf "%-6s %-8s %s\n" "${STATUS[$i]}" "${TIMES[$i]}" "${NAMES[$i]}"
  [[ "${STATUS[$i]}" == "PASS" ]] || fail=1
done
printf "%-6s %-8s %s\n" "" "$(awk -v a="$T_ALL" -v b="$T_END" 'BEGIN{printf "%.1fs", b-a}')" "total"
echo "========================================================"
if [[ $fail -ne 0 ]]; then
  echo "STAGE TESTS FAILED (workdir kept: $WORK, logs: $LOG)"; exit 1
fi
echo "STAGE TESTS PASSED"
if [[ "${CLEANUP_ON_PASS:-0}" == "1" && "${KEEP_WORKDIR:-0}" != "1" ]]; then rm -rf "$WORK"; else echo "workdir kept: $WORK"; fi
