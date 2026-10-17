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
| 10-17 23:11 | baselines agent: download raw data (2 cores, 8 GB, ~3 min) | 0.1 core-h | 0.01 |
| 10-17 23:16 | prep PBMC 68k (4 cores, 16 GB, ~1 min) | 0.1 core-h | 0.01 |
| 10-17 23:16 | prep brain 100k/500k/all (4 cores, 160 GB, ~12 min) | 0.8 core-h | 0.30 |
| 10-17 23:17 | Seurat + Scanpy PBMC 68k, 3 reps (16 cores, 64 GB, ~14 min total) | 3.7 core-h | 0.30 |
| 10-17 23:28 | Seurat + Scanpy brain 100k, 3 reps (16 cores, 64 GB, ~23 min total) | 6.1 core-h | 0.49 |
| 10-17 23:28 | Seurat + Scanpy brain 500k, 1 rep + warm-up (16 cores, 128 GB; Seurat 38 min, Scanpy 20 min) | 15 core-h | ~1.7 |
| 10-17 23:45 | Scanpy brain 1.29M, 1 rep + warm-up (16 cores, 192 GB, 43 min) | 11 core-h | ~1.7 |
| 10-18 00:30 | baselines agent total: `modal billing report` says $3.36 for app scfast-baselines up to 00:00 BST; estimate incl. last 30 min | | ~4.5 |

## Done

- Real data and CPU baselines (`bench/modal_baselines.py`, branch `t3/scfast-real-data-baselines`).
  Modal volume `scfast-data`: `raw/` (10x files, `pbmc68k_labels.tsv` from Zheng et al.),
  `prep/brain/{n100000,n500000,n1290549}`, `prep/pbmc68k/n68260` (with `cells.tsv` labels),
  `results/<dataset>/<engine>_n<cells>/`. Local copies: `bench/results/{pbmc68k_cpu16,brain_cpu16}`.
  QC brain: `--mt-prefix mt- --min-genes 200 --max-genes 6000 --max-mt 20` (1,306,127 → 1,290,549
  cells, 22,788 genes). QC PBMC: defaults (200 < genes < 2500, MT- < 5 %) → 68,260 of 68,579 cells.
  Spec: Modal `cpu=16` (32 logical CPUs), 64 GB (128/192 GB at 500k / all), threads = 16.
  To compare scfast: run it on the same `prep/<dataset>/n<cells>` folder, put its results.jsonl
  lines and `outputs/` into the same local run folder, then `parity.py` + `report.py`.
- Published 1.3M-brain references: `bench/results/REFERENCES.md`.
- CPU baselines (compute = normalize → UMAP, median of measured reps; read separate):

  | Dataset | Cells | Engine | Reps | Compute s | Read s | Slowest step |
  |---|---:|---|---:|---:|---:|---|
  | PBMC 68k | 68,260 | Seurat 5.6.0 | 3 | 121.4 | 4.6 | UMAP 69.5 s |
  | PBMC 68k | 68,260 | Scanpy 1.12.4 | 3 | 57.2 | 2.0 | UMAP 34.5 s |
  | brain | 100,000 | Seurat | 3 | 194.6 | 21.0 | UMAP 106.5 s |
  | brain | 100,000 | Scanpy | 3 | 75.8 | 8.8 | UMAP 49.5 s |
  | brain | 500,000 | Seurat | 1 | 966.8 | 88.4 | UMAP 579 s |
  | brain | 500,000 | Scanpy | 1 | 488.7 | 52.0 | UMAP 325 s |
  | brain | 1,290,549 | Scanpy | 1 | 1161.6 | 101.0 | UMAP 796 s |

  Seurat at 1.29M not run (published ≈ 3 h on 16 cores; REFERENCES.md).

## In progress

## Remaining ideas (ranked)

## Evidence
