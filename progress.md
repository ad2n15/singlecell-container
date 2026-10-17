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

CPU kNN / SNN / Leiden / PCA (worktree t3-110617a5, branch t3/110617a5; bench/cpu_steps.py):
- kNN: exact k-d tree for <= 16 dims (bit-identical to brute force) and `method="kdtree"`
  (leaf cap 64, approximate). 500k Mac: exact GEMM 11.3 s -> kd exact 3.9 s -> kd cap 1.2 s;
  overlap with exact 0.96 (500k), 0.98 (100k); Leiden ARI vs exact-graph labels 0.997.
  HNSW (m 16, ef_c 200) was 8.5 s at 100k: not competitive at 10 dims.
- Leiden: own copy of single-clustering's loop; parallel batched local move + parallel
  per-community refinement + BFS node order; deterministic across thread counts.
  500k Mac: 5.0 s -> 1.8 s; ARI vs leidenalg 0.999 (100k), 0.964 (500k, leidenalg's own
  seed noise; vs planted types scfast 0.965, leidenalg 0.933).
- PCA: register-tiled Z V / Z^T U kernels, bit-identical. 500k x 2000 HVGs: 1.9 s -> 1.05 s.
- SNN untouched (0.55 s at 500k, already parallel).

## In progress

- Modal 16-CPU before/after runs (bench/modal_cpu_steps.py::steps): brain 100k/500k/1.29M
  and synthetic 1M; JSON lands in bench/results/cpu_steps/. Spend so far ~$0.5 (estimate).

## Remaining ideas (ranked)

- Leiden aggregation (single-clustering CSRNetwork::aggregate, serial, 0.3 s at 500k).
- kNN leaf scan in f32 for the capped mode (needs f64 re-rank of the final k).
- Bench engine still calls `method="exact"` (now the k-d tree); switch to `"kdtree"` once the
  Modal overlap/ARI numbers hold on real data.
- Pre-existing failures, not from this work: tests/python test_public_api (umap in __all__),
  umap::tests::two_blobs_separate is flaky under parallel test load.

## Evidence
