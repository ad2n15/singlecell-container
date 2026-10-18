Three runs of the final scfast-cpu code (perf/hillclimb at 107ed9d..9c4ef64) on fresh Modal cpu=16
containers, each `bench/run_py.py --engine scfast-cpu` (warm-up + 2-3 measured repeats). Round 1 has no
CPU record; rounds 2-3 record vendor/family/model in env_r*/ (AMD family 25 model 1 = Milan, model 17 =
Genoa; Intel family 6 model 85/143 = Xeon). Compute medians (normalise..UMAP), s:

| Dataset | round 1 | round 2 | round 3 | median |
|---|---:|---:|---:|---:|
| PBMC 68k | 2.19 | 1.58 | 2.26 | 2.19 |
| brain 100k | 2.81 | 2.43 | 2.58 | 2.58 |
| brain 500k | 11.06 | 11.86 | 9.48 | 11.06 |
| brain 1.29M | 26.85 | 26.68 | 26.37 | 26.68 |
