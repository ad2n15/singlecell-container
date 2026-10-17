# Performance hillclimb (branch `perf/hillclimb`)

Goal: make the core pipeline (read → normalize → HVG → PCA → neighbours → Leiden → UMAP) as fast as
possible, keep the parity gates (ARI ≥ 0.9 vs Seurat and vs planted types; bench/README.md), and
produce a benchmark for a 3-minute demo talk.

Constraints from the user (2026-10-17):
- Modal budget: soft $30 total (raised 23:15). Allocated: GPU agent $8, real-data agent $8, scale agent $4, headline ~$10. L4 for sweeps, one H100 run for the headline.
- Parity gates stay. Commit after each change. Local branch only; no push.
- Modal CLI: `uvx modal` (profile mukeshkbj). No Docker locally. `goalctl` is not installed.

## Spend log (Modal, estimated)

| When | What | GPU-h / CPU-h | Est. $ |
|---|---|---|---|

## Done
- Baseline, Mac M-series 12 threads, 100k synthetic cells, compute s (excl. read): Seurat 87.3, Scanpy 50.3, scfast-cpu 33.9 (UMAP 29.5).
- Rust UMAP (`scfast.umap`, f51a3ba): UMAP 29.5 -> 2.6 s; then latency fixes (6ee2e62, 7716230): 1.3 s at 12 threads, 5.9 s single thread. Quality equal to umap-learn (PBMC 3k silhouette 0.458 vs 0.460).

## In progress
Subagents (T3 delegate_task, own worktrees, branches from perf/hillclimb):
- GPU path on Modal (Sonnet 5.5): bench/modal_app.py, CUDA tests on L4, scfast-gpu engine.
- CPU front end (Sonnet 5.5): read/normalize/HVG/scale.
- Real data + baselines on Modal (Opus 5.5 medium): 1.3M brain, PBMC 68k, Seurat/Scanpy on Modal CPU, REFERENCES.md.
- Scale path (Opus 5.5 medium): kNN (exact is O(n^2)), SNN, Leiden, PCA at 500k-1M.

## Remaining ideas (ranked)

## Evidence
