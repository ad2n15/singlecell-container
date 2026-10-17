# scpipe exchange format — specification v1.0

The contract for moving data between **Seurat (R, CPU)** and **AnnData / rapids-singlecell (Python, GPU)** inside one pipeline run. Both the R and Python implementations must follow this document exactly; the cross-language tests in `tests/integration/` enforce it.

## 1. File format

- An exchange file is a standard **AnnData `.h5ad`** file (HDF5), readable by `anndata.read_h5ad()` with no warnings.
- It follows the AnnData on-disk specification: every group and dataset carries the `encoding-type` and `encoding-version` attributes that AnnData expects.
  - Dataframe: `encoding-type="dataframe"`, `encoding-version="0.2.0"`, with `_index` and `column-order` attributes.
  - CSR matrix: `encoding-type="csr_matrix"`, `encoding-version="0.1.0"`, `shape` attribute, datasets `data`, `indices`, `indptr`.
  - Dense array: `encoding-type="array"`, `encoding-version="0.2.0"`.
  - Categorical: `encoding-type="categorical"`, `encoding-version="0.2.0"`, `ordered` attribute, datasets `codes` (int, −1 = missing) and `categories`.
  - String array: `encoding-type="string-array"`, `encoding-version="0.2.0"`, variable-length UTF-8.
  - Mapping (`uns` and nested dicts): `encoding-type="dict"`, `encoding-version="0.1.0"`.
  - Root group: `encoding-type="anndata"`, `encoding-version="0.1.0"`.
- **Orientation:** AnnData is cells × genes. Seurat is genes × cells, so matrices are transposed on write and read.
- **Sparse matrices** are stored as CSR (cells × genes). The R side writes a Seurat genes × cells `dgCMatrix` (CSC) directly as cells × genes CSR without densifying: the CSC arrays of a genes × cells matrix are exactly the CSR arrays of its transpose.
- **Integer counts** are stored as `int32` when all values fit, otherwise `int64`. Normalised data is `float32`.

## 2. Modes

| Mode | `X` | `layers["counts"]` | Purpose |
|---|---|---|---|
| `full` | log-normalised data, all genes | raw counts, all genes | Moving a whole stage between CPU and GPU |
| `slim` | log-normalised data, **highly variable genes only** | omitted | Sending only what a GPU step needs (scale, PCA, Harmony, neighbours, Leiden, UMAP) |
| `results` | empty (shape cells × 0) | omitted | What the GPU side sends back: embeddings, graphs, cluster labels |

## 3. Slot mapping

| Seurat | AnnData | Rules |
|---|---|---|
| cell barcodes (`colnames`) | `obs_names` (`obs/_index`) | Must be unique; never reordered silently |
| gene names (`rownames`) | `var_names` (`var/_index`) | See section 5 for name protection |
| `counts` layer | `layers["counts"]` | `full` mode only |
| `data` layer | `X` | `full` and `slim` |
| `scale.data` | — | Never transferred; recomputed on the receiving side |
| `meta.data` columns | `obs` columns | factor → categorical (levels and order kept); character → string array; logical → bool; integer → int32/int64; numeric → float64 |
| `VariableFeatures()` | `var["highly_variable"]` (bool) | |
| reductions `pca`, `harmony`, `umap` | `obsm["X_pca"]`, `obsm["X_harmony"]`, `obsm["X_umap"]` | dense float32, cells × dims; other reductions as `obsm["X_<name>"]` |
| PCA feature loadings | `varm["PCs"]` | `full` mode only; genes × dims |
| graphs `<assay>_snn`, `<assay>_nn` | `obsp["connectivities"]`, `obsp["distances"]` | CSR, cells × cells |
| cluster labels | `obs["leiden"]` (categorical) on the GPU side; `seurat_clusters` on the R side | the results file names the column in `uns["exchange"]["cluster_key"]` |

## 4. Provenance: `uns["exchange"]`

Every exchange file carries this mapping. Readers must reject files whose `schema_version` major number they don't support.

| Key | Type | Content |
|---|---|---|
| `schema_version` | string | `"1.0"` |
| `mode` | string | `full`, `slim` or `results` |
| `run_id` | string | Pipeline run identifier |
| `stage` | string | Stage that wrote the file, e.g. `S2_gpu_core` |
| `writer` | string | `"R"` or `"python"` |
| `created_utc` | string | ISO 8601 timestamp |
| `config_hash` | string | SHA-256 of the run config, or `"none"` |
| `n_cells`, `n_genes` | int | Shape of the file |
| `cells_sha256` | string | SHA-256 of the cell barcodes joined with `\n` in file order |
| `genes_sha256` | string | SHA-256 of `var_names` joined with `\n` in file order |
| `counts_checksum` | string | `full` mode only: see section 6 |
| `source_counts_checksum` | string | `slim`/`results` modes: the counts checksum of the object they came from |
| `cluster_key` | string | `results` mode: name of the cluster column in `obs` |
| `packages` | mapping string → string | Versions of the writer's key packages |

## 5. Gene-name protection

Seurat replaces `_` with `-` in feature names. To round-trip exactly:

- `var["gene_symbol_original"]` always holds the original names as received by the pipeline.
- `var["gene_id"]` holds Ensembl IDs when known, else empty strings.
- On reading back into Seurat, the R side uses its own feature names for matching and keeps `gene_symbol_original` as feature metadata.

## 6. Counts checksum

A deterministic, language-independent fingerprint of a counts matrix, computed on the **cells × genes CSR** form after sorting column indices within each row:

```
SHA-256 over, in order:
  shape as two little-endian int64
  indptr  as little-endian int64
  indices as little-endian int64
  data    as little-endian float64
```

R and Python must produce the same hex string for the same matrix. A shared fixture in `tests/fixtures/` pins expected values.

## 7. Merge-back rules (results → Seurat)

1. `schema_version` major is supported, and `mode` is `results`.
2. `source_counts_checksum` equals the counts checksum of the Seurat object being updated. Otherwise the results belong to a different object, and the merge is refused.
3. The results' cells must be a subset of, or equal to, the Seurat object's cells, matched **by barcode, never by position**. Missing cells are an error unless the caller allows subsets.
4. Embeddings are added as Seurat `DimReduc` objects (keys `PC_`, `harmony_`, `UMAP_`), graphs as `Graph` objects, and the cluster column as a factor in `meta.data` and as `Idents()`.
5. Existing slots with the same names are replaced only when the caller passes `overwrite = TRUE`.

## 8. Validation

Both sides expose a validator that checks a file against this spec and returns a list of problems (empty if valid):

- Required `uns["exchange"]` keys for the mode.
- Shapes match `n_cells` and `n_genes`; `cells_sha256` and `genes_sha256` match the file.
- `X`, layers, `obsm`, `varm` and `obsp` shapes are consistent with `obs`/`var`.
- Barcodes unique; `var_names` unique.
- In `full` mode, `counts_checksum` matches `layers["counts"]`.
