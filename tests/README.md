# Tests

The exchange format between Seurat (R) and AnnData (Python) is defined in
[`docs/exchange-spec.md`](../docs/exchange-spec.md). Three layers of tests check it.
Run everything from the repository root.

| Layer | Where | What it checks |
|---|---|---|
| Fixture | `tests/fixtures/` | A small synthetic dataset and the checksums every implementation must reproduce |
| R unit tests | `tests/R/` | `R/exchange/exchange.R` on its own |
| Python unit tests | `tests/python/` | `python/scpipe/` on its own |
| Round-trip | `tests/integration/` | R and Python reading each other's files, end to end |

## Requirements

- R 4.3 or newer with Seurat 5, SeuratObject, hdf5r, Matrix, jsonlite, digest and testthat
- Python 3.12 or newer with anndata, scanpy, h5py, scipy, pandas, pytest, igraph and leidenalg

No GPU is needed: `gpu_core` runs with `--device cpu` in the tests.

## Fixture

`tests/fixtures/pbmc_like/` holds a seeded synthetic dataset of 500 cells and 800 genes
in 10x MatrixMarket form (`matrix.mtx` genes × cells, `features.tsv`, `barcodes.tsv`),
cell annotations in `cells.tsv` (two samples, four cell types with their own marker genes),
and `expected.json`, which pins:

- `counts_checksum` (spec section 6), computed by an implementation inside
  `make_fixture.py` that does not use scpipe
- `cells_sha256`, plus `genes_sha256_original` and `genes_sha256_seurat` (spec section 4),
  for the original names and for the names after Seurat turns `_` into `-`
- the order of the factor levels, the genes with underscores, the MT- genes and the marker genes

The gene names include underscores, dashes, MT- genes and look-alike names such as
`TBCE` / `TBCE.1` / `TBCE-AS1`. They stay unique after Seurat renames them.

```bash
python tests/fixtures/make_fixture.py           # regenerate (only if you change the generator)
python tests/fixtures/make_fixture.py --check   # regenerate in a temp dir and compare byte for byte
python tests/fixtures/make_fixture.py --verify  # recompute expected.json from the committed files
```

The committed files are what the tests use. If you change the generator, regenerate the files
and commit them together with it.

## R unit tests

```bash
Rscript -e 'testthat::test_dir("tests/R")'
# or a single file
Rscript -e 'testthat::test_file("tests/R/test-exchange.R")'
```

## Python unit tests

```bash
PYTHONPATH=python pytest -q tests/python
```

## Round-trip

```bash
tests/integration/run_roundtrip.sh              # temp workdir, removed if everything passes
tests/integration/run_roundtrip.sh /tmp/rt      # keep the .h5ad files and logs in /tmp/rt
KEEP_WORKDIR=1 tests/integration/run_roundtrip.sh
```

Set `RSCRIPT` or `PYTHON` to use other interpreters. Each step's log goes to `<workdir>/logs/`.

| Step | Script | What it does |
|---|---|---|
| S0 | `make_fixture.py --check` | The fixture is reproducible |
| S1 | `roundtrip.R write` | Loads the fixture into Seurat, runs NormalizeData and FindVariableFeatures (500 genes), checks `sc_counts_checksum` against `expected.json`, writes `r_full.h5ad` and `r_slim.h5ad`, and checks that `validate_exchange` returns no problems |
| S2 | `roundtrip.py check-r-files` | Reads both files with `anndata.read_h5ad` (any warning fails the step) and with `scpipe.exchange.read_exchange`. Checks the encodings, `uns["exchange"]`, the counts checksum, that the slim file holds only the variable genes, that factor levels are kept, and that `var["gene_symbol_original"]` keeps the underscores |
| S3 | `python -m scpipe.gpu_core ... --device cpu` | Turns the slim file into `results.h5ad` |
| S3b | `roundtrip.py check-results` | Checks the results file (`obsm`, `obsp`, cluster column, and that the clusters match the planted cell types). Builds the files for the refuse cases: one with an altered source checksum, and one missing a barcode |
| S4 | `roundtrip.R merge` | Merges the results into the original object. Checks the reductions, keys, graphs, clusters in `meta.data` and `Idents()`, and the cell order, then merges into a shuffled object to check matching by barcode. Refuse cases: merging twice without `overwrite`, a wrong checksum, a missing barcode (accepted with `allow_subset = TRUE`), and an unknown barcode |
| S5a | `roundtrip.py write-python` | Python builds a full file from the fixture (counts plus log1p `X`) with `write_exchange` |
| S5b | `roundtrip.R read-python` | `read_exchange_seurat` reads the Python file. Checks the counts checksum, factor levels, column types, feature metadata and variable features |

The script prints a PASS/FAIL table with timings at the end. It exits with a non-zero code if any
step fails, or is skipped because a step it depends on failed.

## CI

`.github/workflows/exchange-tests.yml` runs all of the above on `ubuntu-latest` for pushes
and pull requests that touch `R/`, `python/`, `tests/` or `docs/exchange-spec.md`.
If the round-trip fails, its workdir is uploaded as a build artifact.
