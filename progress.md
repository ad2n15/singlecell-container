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
| 10-17 22:40-23:50 UTC | GPU agent: image build, wheel builds, CUDA tests, L4 benches and profiling (app `scfast-bench`, Modal billing report) | ~1.3 L4-h + builds | 2.06 |
| 10-18 00:00-02:10 UTC | GPU agent: L4 profiling, final L4 run (4 engines), H100 runs 500k / 1M (app `scfast-bench`, billing report) | ~2 L4-h, ~0.4 H100-h | ~4.1 |

## Done
- Baseline, Mac M-series 12 threads, 100k synthetic cells, compute s (excl. read): Seurat 87.3, Scanpy 50.3, scfast-cpu 33.9 (UMAP 29.5).
- Rust UMAP (`scfast.umap`, f51a3ba): UMAP 29.5 -> 2.6 s; then latency fixes (6ee2e62, 7716230): 1.3 s at 12 threads, 5.9 s single thread. Quality equal to umap-learn (PBMC 3k silhouette 0.458 vs 0.460).

## GPU path (branch t3/837279cd, GPU agent round 2) - done
Compute seconds (excl. read), 1 measured rep after a warm-up, synthetic 10-type data (bench/synth.py + prep.py).
L4 box: 8 CPU cores, 48 GB. H100 box: 8 CPU cores, 160 GB. Results in bench/results/{l4_final,h100_500k,h100_1m} (git-ignored; also on volume `scfast-bench`).

| engine | GPU | cells | to_gpu | norm | hvg | scale | pca | neighbors | cluster | umap | to_cpu | compute |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scanpy | (L4 box CPU) | 100k | - | 0.99 | 2.34 | 3.61 | 33.20 | 9.46 | 5.97 | 87.15 | - | 142.72 |
| scfast-cpu | (L4 box CPU) | 100k | - | 0.96 | 2.29 | 2.34 | 0.46 | 1.43 | 1.99 | 11.19 | - | 20.66 |
| rapids | L4 | 100k | 0.22 | 0.03 | 0.05 | 0.07 | 0.27 | 0.30 | 0.54 | 0.23 | 0.36 | 2.07 |
| scfast-gpu | L4 | 100k | 0.22 | 0.03 | 0.05 | 0.04 | 0.20 | 0.27 | 0.62 | 0.24 | 0.07 | 1.74 |
| scanpy | (L4 box CPU) | 500k | - | 4.65 | 11.09 | 16.90 | 232.83 | 62.23 | 37.99 | 485.02 | - | 850.71 |
| scfast-cpu | (L4 box CPU) | 500k | - | 4.71 | 11.12 | 11.16 | 1.99 | 27.19 | 13.15 | 46.42 | - | 115.74 |
| rapids | L4 | 500k | 1.10 | 0.14 | 0.17 | 0.36 | 1.17 | 3.96 | 3.41 | 1.29 | 1.77 | 13.36 |
| scfast-gpu | L4 | 500k | 0.63 | 0.14 | 0.17 | 0.21 | 0.65 | 1.57 | 3.19 | 1.31 | 0.32 | 8.20 |
| rapids | H100 | 500k | 0.69 | 0.02 | 0.08 | 0.13 | 0.23 | 1.35 | 1.19 | 0.40 | 1.40 | 5.49 |
| scfast-gpu | H100 | 500k | 0.56 | 0.02 | 0.07 | 0.08 | 0.31 | 0.65 | 0.72 | 0.38 | 0.26 | 3.05 |
| rapids | H100 | 1M | 2.51 | 0.04 | 0.13 | 0.27 | 0.43 | 4.23 | 1.60 | 0.32 | 2.68 | 12.21 |
| scfast-gpu | H100 | 1M | 1.51 | 0.04 | 0.13 | 1.11 | 0.42 | 1.93 | 1.27 | 0.30 | 0.29 | 7.00 |

Read (scanpy read_10x_h5, all engines): ~8 s at 100k, ~39 s at 500k, ~55-68 s at 1M. It dwarfs GPU compute.

Parity (bench/parity.py): vs scanpy at 100k every engine ARI 1.0, PC similarity >= 0.9999. scfast kNN overlap vs scanpy 0.94
(both scfast engines, from the shared unclipped PCA), rapids 0.999. ARI vs truth at 500k: rapids 1.0, scfast-cpu 0.93 (11
clusters), scfast-gpu 0.85 (12); H100 1M scfast-gpu 0.93 (11). The 500k split is Seurat's objective: directed RB quality at
res 0.5 on the 500k SNN is truth 2,696,230 < CPU Leiden (11) 2,697,092 < cuGraph (12) 2,698,180. Open decision: accept,
or scale resolution with n. (Diagonal at half weight in cuGraph under-splits at res 1 and 2: 100k 13/20 vs CPU 14/22.)

What changed (commits 99f1ee0..3b74cba):
- Image: prebuilt `rapids-singlecell-cu12` 0.18.1 + RAPIDS 26.10 wheels (no CMake). Modal: touch mounted sources so cargo
  rebuilds; `script` entrypoint runs a profiling script on the GPU box.
- CUDA bugs: cuGraph dtype enum (INT32 = 2, not 0); Leiden max_level = n_iterations (2) gave 580 fragments, now 100 (rapids
  engine had the same bug via rsc); kNN `topk` O(n^2) with L1 thrash; serial symmetry check before cuGraph (2 s at 500k).
- kNN: fused kernel (|r|^2 - 2 q.r, 4 queries per thread, NVRTC per dims/cap), Morton order, tile pruning by bounding box.
- PCA: Gram path (f64-atomic X^T X on device, subspace iteration on host); takes a CuPy CSR (device copy, colstats kernel).
- SNN with cuSPARSE from CuPy indices; Leiden on cuGraph device views; UMAP fuzzy graph with CuPy (= cuML to 4e-6).
- Both GPU engines: cudaMallocAsync via RMM (RMM's fixed pool ran out at 500k on the L4).
- Rust UMAP vs cuML: cuML (0.24 s at 100k on L4) kept; Rust UMAP is 11.7 s on the 8-core Modal box (1.3 s on the Mac; unexplained).

Reproduce:
    uvx modal run bench/modal_app.py::build --no-tests
    uvx modal run bench/modal_app.py::rust_tests                      # all CUDA / cuGraph tests on an L4
    uvx modal run bench/modal_app.py::gen --tag synth --sizes 100000,500000
    uvx modal run bench/modal_app.py::gen --tag synth_1m --sizes 1000000
    uvx modal run bench/modal_app.py::bench --tag synth --name l4_final --sizes 100000,500000 --engines scfast-gpu,rapids,scfast-cpu,scanpy
    SCFAST_GPU=H100 SCFAST_MEM_GB=160 uvx modal run bench/modal_app.py::bench --tag synth --name h100_500k --sizes 500000 --engines scfast-gpu,rapids
    SCFAST_GPU=H100 SCFAST_MEM_GB=160 uvx modal run bench/modal_app.py::bench --tag synth_1m --name h100_1m --sizes 1000000 --engines scfast-gpu,rapids
    uv run --no-project --with numpy --with pandas --with scikit-learn --with scipy python bench/parity.py --run bench/results/l4_final --reference scanpy

Remaining ideas: cuGraph Leiden is the floor of `cluster` (3.0 s at 500k on L4, renumbering does not matter); kNN host parts
(Morton order, re-rank) at 1M; read path (CPU agent).

## In progress
Subagents (T3 delegate_task, own worktrees, branches from perf/hillclimb):
- GPU path on Modal (Sonnet 5.5): bench/modal_app.py, CUDA tests on L4, scfast-gpu engine.
- CPU front end (Sonnet 5.5): read/normalize/HVG/scale.
- Real data + baselines on Modal (Opus 5.5 medium): 1.3M brain, PBMC 68k, Seurat/Scanpy on Modal CPU, REFERENCES.md.
- Scale path (Opus 5.5 medium): kNN (exact is O(n^2)), SNN, Leiden, PCA at 500k-1M.

## Remaining ideas (ranked)

## Evidence
