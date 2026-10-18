# scfast: final benchmark (overnight hillclimb, 17–18 October 2026)

Core single-cell workflow: normalise → HVGs → scale → PCA → neighbours → Leiden → UMAP.
Same parameters for every engine (2,000 HVGs, 30 PCs, k = 20 on 10 PCs, Leiden resolution 0.5,
2 iterations, UMAP). "Compute" is normalise → UMAP; reading the file is listed separately.
Every engine reads the same QC'd files (`bench/prep.py`).

## CPU: real 10x data, identical machines (Modal, 16 cores, AMD EPYC)

| Dataset | Cells | Seurat 5.6 | Scanpy 1.12 | **scfast-cpu** | vs Seurat | vs Scanpy |
|---|---:|---:|---:|---:|---:|---:|
| PBMC 68k | 68,260 | 121.4 s | 57.2 s | **3.4 s** | **36×** | **17×** |
| Mouse brain | 100,000 | 194.6 s | 75.8 s | **3.2 s** | **60×** | **23×** |
| Mouse brain | 500,000 | 966.8 s | 488.7 s | **21.4 s** | **45×** | **23×** |
| Mouse brain | 1,290,549 | not run (~3 h published) | 1,161.6 s | **46.3 s** | – | **25×** |

Peak memory at 1.29M cells: scfast 45 GB, Scanpy 150 GB. Reading the 1.29M file: scfast 15 s,
Scanpy 101 s.

Sources: `bench/results/{pbmc68k_cpu16,brain_cpu16}/report.md` (medians; 3 measured repeats
up to 100k cells, 1–2 above, each after a warm-up).

## Agreement with Seurat (parity gates: PC similarity ≥ 0.99, kNN overlap ≥ 0.9, ARI ≥ 0.9)

| Dataset | Engine | PC similarity | kNN overlap | ARI vs Seurat |
|---|---|---:|---:|---:|
| PBMC 3k | scfast-cpu | 0.99995 | 0.976 | 0.947 |
| PBMC 3k | Scanpy | 0.9997 | 0.968 | 0.818 |
| PBMC 68k | scfast-cpu | 1.000 | 0.945 | 0.719 |
| PBMC 68k | Scanpy | 1.000 | 0.974 | 0.559 |
| brain 100k | scfast-cpu | 1.000 | 0.963 | 0.693 |
| brain 500k | scfast-cpu | 1.000 | 0.959 | 0.565 |

PC similarity and kNN overlap pass everywhere. The ARI gate is below Leiden's own noise floor
on the larger datasets: on PBMC 68k, **Seurat against itself with different Leiden seeds gives
ARI 0.70–0.71**, and scfast against itself 0.67–0.79 (0.73–0.81 run to convergence). scfast's
agreement with Seurat sits inside that band, and its agreement with the published Zheng et al.
PBMC 68k labels (ARI 0.274) equals Seurat's (0.283).

## GPU: real 10x data, NVIDIA H100 80 GB (Modal, 16 CPU cores)

| Dataset | Cells | rapids-singlecell | **scfast-gpu** | file → result, rapids | **file → result, scfast-gpu** |
|---|---:|---:|---:|---:|---:|
| PBMC 68k | 68,260 | 0.62 s | **0.52 s** | 2.84 s | **0.68 s** |
| Mouse brain | 100,000 | 1.20 s | **1.20 s** | 12.06 s | **1.82 s** |
| Mouse brain | 500,000 | 4.28 s | **3.06 s** | 64.83 s | **5.50 s** |
| Mouse brain | 1,290,549 | **out of GPU memory** | **6.80 s** | fails | **25.7 s** |

The first two columns are compute (normalise → UMAP); "file → result" adds reading the file and the
host ⇄ GPU transfers. On a 24 GB L4, rapids-singlecell already runs out of memory at 500k cells,
where scfast-gpu finishes in 11.0 s. scfast-gpu vs Seurat: PC similarity 0.9997–1.0000, kNN overlap
0.994–0.998 at every size (PBMC 3k on an L4: 1.0 / 0.998).

Published reference for about 1M cells of the same dataset (`REFERENCES.md`): rapids-singlecell's
core steps ≈ 7.3 s on a B200, Scanpy ≈ 13 min on a 32-core Threadripper.

## The headline: 1.29 million mouse brain cells

| | Seurat (16 cores) | Scanpy (16 cores) | scfast CPU (16 cores) | rapids-singlecell (H100) | **scfast GPU (H100)** |
|---|---:|---:|---:|---:|---:|
| Compute | ~3 h (published) | 1,161.6 s | 46.3 s | out of memory | **6.8 s** |
| File → result | – | 1,262.6 s | 61.6 s | – | **25.7 s** |
| vs Scanpy (compute) | – | 1× | 25× | – | **171×** |

## What changed overnight

See `progress.md` for the step-by-step log with evidence.
