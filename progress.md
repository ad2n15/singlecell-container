# Performance hillclimb (branch `perf/hillclimb`)

Goal: make the core pipeline (read → normalize → HVG → PCA → neighbours → Leiden → UMAP) as fast as
possible, keep the parity gates (ARI ≥ 0.9 vs Seurat and vs planted types; bench/README.md), and
produce a benchmark for a 3-minute demo talk.

Constraints from the user (2026-10-17):
- Modal budget about $20–25 total. L4 for sweeps, one H100 run for the headline.
- Parity gates stay. Commit after each change. Local branch only; no push.
- Modal CLI: `uvx modal` (profile mukeshkbj). No Docker locally. `goalctl` is not installed.

## Spend log (Modal, estimated)

| When | What | GPU-h / CPU-h | Est. $ |
|---|---|---|---|

## Done

CPU kNN / SNN / Leiden / PCA (branch t3/110617a5; bench/cpu_steps.py, bench/modal_cpu_steps.py).
Per-step medians of 3 (s). before = f51a3ba (GEMM exact kNN, single-clustering Leiden);
after = HEAD with `method="auto"` kNN (k-d tree, eps 0.7, above 50k cells).

| data | machine | PCA | kNN | SNN | Leiden | sum |
|---|---|---|---|---|---|---|
| synth 100k | Mac 12-core, before | 0.42 | 0.51 | 0.07 | 0.78 | 1.78 |
| synth 100k | Mac 12-core, after | 0.24 | 0.10 | 0.07 | 0.31 | 0.72 |
| synth 500k | Mac 12-core, before | 2.07 | 11.26 | 0.55 | 5.05 | 18.93 |
| synth 500k | Mac 12-core, after | 1.07 | 0.83 | 0.45 | 1.56 | 3.91 |
| brain 1.29M | Mac 12-core, after | 1.91 | 4.00 | 1.37 | 5.07 | 12.35 |
| brain 100k | Modal 16 CPU, before | 0.16 | 15.51 | 0.08 | 1.01 | 16.76 |
| brain 100k | Modal 16 CPU, after | 0.16 | 0.14 | 0.09 | 1.02 | 1.41 |
| brain 500k | Modal 16 CPU, before | 0.88 | 179.46 | 0.59 | 12.16 | 193.09 |
| brain 500k | Modal 16 CPU, after | 0.59 | 0.95 | 0.41 | 2.99 | 4.94 |
| synth 1M | Modal 16 CPU, before | 1.87 | 28.71 | 0.96 | 15.09 | 46.63 |
| synth 1M | Modal 16 CPU, after | 2.09 | 2.60 | 1.30 | 7.36 | 13.35 |
| brain 1.29M | Modal 16 CPU, after | 1.61 | 3.98 | 1.68 | 10.11 | 17.38 |

Raw JSON: bench/results/cpu_steps/ in the t3-110617a5 worktree (gitignored, local only):
modal16_{before,after,final_after}_*.json, mac12_after_*.json.
Brain 1.29M "before" was stopped (GEMM exact kNN ~20 min per repeat by n^2 from 500k).
Modal "after" rows are HEAD eee161a; Mac rows HEAD 148d418+ (same code paths).

Approximation adopted: kNN eps 0.7 k-d tree. Overlap with exact neighbours: 0.957 (synth
100k), 0.955 (synth 500k), 0.951 (synth 1M), 0.963 / 0.962 / 0.946 (brain 100k / 500k /
1.29M). Leiden ARI against labels from the exact graph: 0.998 / 0.996 / 0.999 (synth 100k /
500k / 1M); vs planted types unchanged (0.953 / 0.964 / 0.949). On brain, Leiden itself is
unstable at resolution 0.5: seed-to-seed ARI on the exact graph is 0.55-0.58, and the
approximate graph's ARI (0.56-0.59) sits inside that noise, so ARI cannot judge kNN there.
Leiden changes vs single-clustering: ARI vs leidenalg 0.999 (synth 100k).

What changed (all deterministic for any thread count; bit-identical where stated):
- kNN: exact k-d tree for <= 16 dims (bit-identical to brute force), eps-approximate mode,
  points/boxes in f32 widened per use. HNSW was 16x slower than exact at 100k (10 dims).
- Leiden: own copy of single-clustering's loop; parallel batched local move (validated
  moves), per-community parallel refinement, parallel aggregation (identical graph), BFS
  node order.
- PCA: register-tiled Z V / Z^T U (bit-identical): 1.8x on Apple silicon, neutral on Modal x86.
- SNN: rows in BFS order of the kNN graph (bit-identical): 1.70 -> 1.36 s at 1.29M.

Tried and dropped: f32 inner power iterations in PCA (5% after tiling); fixed leaf cap for
the k-d tree (recall fell with n); Leiden apply shortcuts via speculative weights (fallback
54% of moves) and via corrections (net 7%, not bit-identical).

## In progress

Nothing running. Modal spend for this work: about $2.5 (estimate: inputs ~12 min of up to
4 x 8-core/96 GB, step runs ~27 + ~10 min of up to 8 x 16-core/48 GB, one 16-core tile test).

## Remaining ideas (ranked)

- Leiden level-0 local move is the largest step at 1M+ (Modal x86 7-10 s): its serial
  apply re-scans each moved node's edges (1.5 s of 5 s on the Mac at 1.29M brain).
- kNN: share traversal across a leaf's queries, or an f32 leaf scan for the eps mode
  (search is now 36% leaf scan, 25% loop, 19% box bounds, 14% heap).
- Pre-existing failures, not from this work: tests/python test_public_api (umap in __all__),
  umap::tests::two_blobs_separate is flaky under parallel test load.

## Evidence
