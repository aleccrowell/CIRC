# circ.limbr — KNN imputation and SVA batch-effect removal

LIMBR processes raw proteomics or RNAseq expression matrices in two stages:

1. **Imputation** — deduplicates peptides, filters rows with too many missing values, and fills remaining gaps with K-nearest neighbours.
2. **Batch-effect removal** — applies pool normalization (proteomics), quantile normalization, and SVA-based identification and removal of latent batch effects.

## CLI

### `circ impute`

```
circ impute -f INPUT -o OUTPUT [-m MISSINGNESS] [-n NEIGHBORS]
```

| Flag | Default | Description |
|---|---|---|
| `-f/--filename` | required | Input TSV (Peptide/Protein or # index + ZT columns) |
| `-o/--output` | required | Output file path |
| `-m/--missingness` | 0.3 | Maximum allowable missing fraction per row |
| `-n/--neighbors` | 10 | Number of nearest neighbours for KNN imputation |

### `circ normalize`

```
circ normalize -f INPUT -d DESIGN -t DATATYPE -o OUTPUT [options]
```

| Flag | Default | Description |
|---|---|---|
| `-f/--filename` | required | Imputed input file (TSV or Parquet) |
| `-d/--design` | required | `c` = circadian, `t` = timecourse, `b` = block |
| `-t/--data-type` | required | `p` = proteomics (Peptide/Protein), `r` = RNAseq (#) |
| `-b/--blocks` | None | Parquet file with `block` column (required for `b` design) |
| `-p/--pool` | None | Parquet file with `pool_number` column (proteomics pools) |
| `--nperm` | 200 | Permutation iterations for batch effect significance |
| `--nproc` | 1 | Parallel worker processes |
| `-o/--output` | required | Output file path |

## Python API

### `circ.limbr.imputation.imputable`

```python
from circ.limbr.imputation import imputable

# Accepts a file path (TSV or Parquet) or a DataFrame
obj = imputable("raw.txt", missingness=0.3, neighbors=10)
obj.impute_data("imputed.parquet")   # extension determines output format
```

Deduplicates peptides, drops rows exceeding the missingness threshold, and
imputes remaining missing values using K-nearest neighbours.

### `circ.limbr.batch_fx.sva`

```python
from circ.limbr.batch_fx import sva

# Accepts a file path (TSV or Parquet) or a DataFrame
obj = sva("imputed.parquet", design="c", data_type="p", pool="pool_map.parquet")
obj.preprocess_default()
obj.perm_test(nperm=200, npr=1)
obj.output_default("normalized.parquet")
# Diagnostic sidecars written alongside: normalized_trends.parquet,
# normalized_perms.parquet, normalized_tks.parquet, normalized_pep_bias.parquet
```

Applies pool normalization (proteomics), quantile normalization, and SVA-based
identification and removal of latent batch effects.

`design` options: `"c"` (circadian), `"t"` (timecourse), `"b"` (blocked).
`data_type` options: `"p"` (proteomics — Peptide/Protein columns), `"r"` (RNAseq — # column).
