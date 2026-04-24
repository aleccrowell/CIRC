# CIRC — Circadian Integrated Research Core

CIRC is a unified Python toolkit for circadian genomics and proteomics analysis,
integrating four complementary tools:

| Module | Purpose |
|---|---|
| `circ.limbr` | KNN imputation + SVA-based batch effect removal |
| `circ.pirs` | Prediction Interval Ranking Score for constitutive expression |
| `circ.bootjtk` | Bootstrap empirical JTK circadian rhythm detection |
| `circ.expression_classification` | Unified classifier combining PIRS and BooteJTK |

A single `circ` CLI entry point wraps all three core tools, and all modules share
the same `ZT{HH}_{rep}` column format so outputs chain directly between steps.

All modules accept either a **file path** or a **pandas DataFrame** as input,
and all file I/O supports both **Apache Parquet** (`.parquet`) and tab-separated
text (`.txt` / `.tsv`).  The shared `circ.io` utilities handle format detection
automatically by file extension.

## Installation

Requires Python 3.11+. Install with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/aleccrowell/CIRC.git
cd CIRC
poetry install
```

For optional Numba acceleration of BooteJTK:

```bash
poetry install --extras fast
```

## Data format

All tools expect expression data with samples as columns named `ZT{HH}_{rep}`
(zero-padded two-digit Zeitgeber Time, underscore, replicate number):

```
#       ZT02_1  ZT02_2  ZT04_1  ZT04_2  ZT06_1  ZT06_2
gene_a  1.23    1.19    0.95    1.01    0.88    0.92
gene_b  ...
```

For proteomics data the first two columns are `Peptide` and `Protein` instead
of the single `#` index column.

### File format

Inputs and outputs can be **Apache Parquet** (`.parquet`) or **tab-separated
text** (any other extension).  The format is detected automatically by file
extension everywhere — CLI flags, Python API, and intermediate files all
follow the same rule.  Parquet is preferred for data at rest because it is
faster to read, smaller on disk, and preserves dtypes without round-trip loss.

```bash
# TSV output (backward-compatible)
circ rank -f expr.txt -o ranked_scores.txt

# Parquet output
circ rank -f expr.parquet -o ranked_scores.parquet
```

All modules also accept a **pandas DataFrame** directly in the Python API,
removing the need to write intermediate files when composing steps in a script:

```python
import pandas as pd
from circ.pirs.rank import ranker

df = pd.read_parquet("expr.parquet")
r = ranker(df, anova=False)       # DataFrame passed directly
ranked = r.pirs_sort()
```

## Full pipeline

A typical proteomics circadian experiment using Parquet throughout:

```python
from circ.limbr import simulations, imputation, batch_fx
from circ.expression_classification.classify import Classifier

# 1. Simulate (or start from real data)
sim = simulations.simulate(tpoints=12, nrows=500, nreps=2, rseed=42)
sim.generate_pool_map(out_name="pool_map")
sim.write_output(out_name="sim")

# 2. Impute missing values — write Parquet for fast downstream reading
imp = imputation.imputable("sim_with_noise.txt", missingness=0.3, neighbors=10)
imp.impute_data("imputed.parquet")

# 3. Remove batch effects — reads Parquet, writes Parquet
#    Diagnostic sidecars (imputed_trends.parquet, _perms.parquet, etc.) are
#    written alongside the main output with a matching extension.
sva_obj = batch_fx.sva("imputed.parquet", design="c", data_type="p",
                        pool="pool_map.parquet")
sva_obj.preprocess_default()
sva_obj.perm_test(nperm=200)
sva_obj.output_default("normalized.parquet")

# 4. Classify expression patterns — accepts the Parquet path directly
clf = Classifier("normalized.parquet", reps=2)
result = clf.run_all(slope_pvals=True, n_permutations=1000, n_jobs=4)
# result.label: constitutive | rhythmic | linear | variable | noisy_rhythmic

# Constitutive genes make good normalization references
constitutive = result[result["label"] == "constitutive"].index
```

Modules can also be composed **in-memory** by passing DataFrames directly —
no intermediate files required:

```python
import pandas as pd
from circ.limbr.batch_fx import sva
from circ.expression_classification.classify import Classifier

# Load once, pass by reference
df = pd.read_parquet("imputed.parquet")

sva_obj = sva(df, design="c", data_type="p", pool="pool_map.parquet")
sva_obj.preprocess_default()
sva_obj.perm_test(nperm=200)
sva_obj.output_default("normalized.parquet")

normalized = sva_obj.svd_norm   # DataFrame available directly after normalize()

clf = Classifier(normalized, reps=2)
result = clf.run_all(slope_pvals=True, n_permutations=1000, n_jobs=4)
```

For finer control over the PIRS step (e.g. writing ranked scores to disk):

```python
from circ.pirs.rank import ranker

r = ranker("normalized.parquet", anova=True)
r.get_tpoints()
ranked = r.pirs_sort(
    outname="ranked_scores.parquet",
    slope_pvals=True,
    n_permutations=1000,
    n_jobs=4,
)
```

## CLI reference

```
circ COMMAND [options]
```

| Command | Tool | Description |
|---|---|---|
| `impute` | LIMBR | KNN imputation of missing data |
| `normalize` | LIMBR | SVA batch effect removal |
| `rank` | PIRS | Rank profiles by constitutiveness |
| `rhythm` | BooteJTK | Bootstrap JTK rhythm detection |
| `rhythm-calcp` | BooteJTK | Bootstrap JTK + CalcP full pipeline |

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
| `-f/--filename` | required | Imputed input TSV |
| `-d/--design` | required | `c` = circadian, `t` = timecourse, `b` = block |
| `-t/--data-type` | required | `p` = proteomics (Peptide/Protein), `r` = RNAseq (#) |
| `-b/--blocks` | None | Parquet file with `block` column (required for `b` design) |
| `-p/--pool` | None | Parquet file with `pool_number` column (proteomics pools) |
| `--nperm` | 200 | Permutation iterations for batch effect significance |
| `--nproc` | 1 | Parallel worker processes |
| `-o/--output` | required | Output file path |

### `circ rank`

```
circ rank -f INPUT -o OUTPUT [--no-anova]
```

| Flag | Default | Description |
|---|---|---|
| `-f/--filename` | required | Normalized input TSV (# index + ZT columns) |
| `-o/--output` | required | Output scores file |
| `--no-anova` | off | Skip ANOVA pre-filtering of rhythmic profiles |

### `circ rhythm` and `circ rhythm-calcp`

These subcommands pass all arguments through to BooteJTK's own parsers.
Run `circ rhythm --help` or `circ rhythm-calcp --help` for the full flag list.

Quick example:

```bash
# Single-step rhythm detection
circ rhythm -f normalized.txt -x OUTPUT_PREFIX -r 2 -z 10

# Full pipeline with empirical p-value calculation
circ rhythm-calcp -f normalized.txt -x OUTPUT_PREFIX -r 2 -z 10
```

Key BooteJTK flags:

| Flag | Description |
|---|---|
| `-f FILE` | Input file |
| `-x PREFIX` | Output file prefix |
| `-r INT` | Number of bootstrap resampling iterations |
| `-z INT` | Number of permutation iterations for empirical p-values |
| `-p PERIODS` | Comma-separated list of periods to test (default: 24) |
| `-a ASYMMETRIES` | Asymmetry values to test |
| `-s PHASES` | Phase offsets to test |

## Module API

### `circ.io` — shared I/O utilities

```python
from circ.io import read_expression, write_expression, sidecar_path

# Read from Parquet or TSV; or pass a DataFrame directly
df = read_expression("data.parquet")          # Parquet
df = read_expression("data.txt")              # TSV, RNAseq (# index)
df = read_expression("data.txt", data_type="p")  # TSV, proteomics (Peptide/Protein)
df = read_expression(existing_df)             # DataFrame passthrough

# Write to Parquet or TSV (format from extension)
write_expression(df, "out.parquet")
write_expression(df, "out.txt")

# Derive a sidecar path that preserves the main output's extension
sidecar_path("out.parquet", "_trends")   # → "out_trends.parquet"
sidecar_path("out.txt", "_perms")        # → "out_perms.txt"
```

These utilities are used internally by all modules, but can also be called
directly when building custom pipelines.

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

### `circ.pirs.rank.ranker`

```python
from circ.pirs.rank import ranker

# Accepts a file path (TSV or Parquet) or a DataFrame
r = ranker("normalized.parquet", anova=True)
r.get_tpoints()
scores = r.calculate_scores()                 # returns DataFrame sorted ascending
ranked = r.pirs_sort(outname="scores.parquet")  # writes Parquet; returns sorted data
```

Scores each expression profile using 95% prediction interval bounds from a
linear regression fit.  For each gene the score is:

```
max over fine time grid of max(|PI_upper(t) − mean_expr|, |PI_lower(t) − mean_expr|)
────────────────────────────────────────────────────────────────────────────────────
                              |mean_expr|
```

A flat, low-noise gene has tight bounds near mean expression everywhere →
score near 0.  A trending or noisy gene has bounds that deviate far from mean
expression → large score.  Lower score = more constitutive.

When `anova=True` (default), rhythmic profiles are removed by one-way ANOVA
before scoring so that constitutive candidates are not contaminated by
circadian genes.

#### Permutation p-values

Two independent permutation tests are available after `calculate_scores()`:

```python
r.calculate_pvals(n_permutations=1000, n_jobs=1)
# Adds columns: pval, pval_bh
# Left-tail test: small p = gene has statistically significant temporal
# structure (good linear fit is more structured than chance).

r.calculate_slope_pvals(n_permutations=1000, n_jobs=1)
# Adds columns: slope_pval, slope_pval_bh
# Right-tail test: small p = gene has a statistically significant linear
# slope (slope collapses toward zero after permutation).
```

Both can be requested through `pirs_sort`:

```python
ranked = r.pirs_sort(
    outname="scores.txt",
    pvals=True,
    slope_pvals=True,
    n_permutations=1000,
    n_jobs=4,
)
```

The two tests are complementary.  `pval` detects clean temporal trend (good
linear fit with low residuals); `slope_pval` detects any significant linear
slope regardless of noise level.  Together they distinguish:

| pval | slope_pval | Interpretation |
|---|---|---|
| large | large | constitutive or rhythmic — no linear trend |
| small | small | clean linear trend |
| large | small | noisy linear trend |

### `circ.expression_classification.classify.Classifier`

Combines PIRS scores and BooteJTK rhythmicity results to classify every gene
into one of five expression patterns:

| Label | Description |
|---|---|
| `constitutive` | Stable expression, no significant slope, not rhythmic |
| `rhythmic` | Stable-to-moderate expression with a strong circadian rhythm |
| `linear` | Significant linear slope, not rhythmic |
| `variable` | High PIRS score (non-constitutive), no slope, not rhythmic |
| `noisy_rhythmic` | High PIRS score with a detectable rhythmic signal |

`linear` is only emitted when slope p-values have been computed (see below).
Without slope p-values the original four-label scheme is used.

```python
from circ.expression_classification.classify import Classifier

# Accepts a file path (TSV or Parquet) or a DataFrame
clf = Classifier("normalized.parquet", anova=False, reps=2, size=50, workers=1)

# Step-by-step
clf.run_pirs(pvals=True, slope_pvals=True, n_permutations=1000, n_jobs=4)
clf.run_bootjtk()
result = clf.classify(
    pirs_percentile=50,       # genes at/below this PIRS percentile are "stable"
    slope_pval_threshold=0.05,
    tau_threshold=0.5,
    emp_p_threshold=0.05,
)

# Or in a single call
result = clf.run_all(
    slope_pvals=True,
    n_permutations=1000,
    n_jobs=4,
)
```

`result` is a DataFrame indexed by gene ID with columns `pirs_score`,
`slope_pval` / `slope_pval_bh` (when computed), `tau_mean`, `emp_p`,
`period_mean`, `phase_mean`, and `label`.

When `slope_pvals=False` (default), `run_pirs` computes only the raw PIRS
score and `classify` falls back to the four-label scheme.  This keeps the
default path fast for exploratory use.

### `circ.pirs.rank.rsd_ranker`

```python
from circ.pirs.rank import rsd_ranker

# Accepts a file path (TSV or Parquet) or a DataFrame
r = rsd_ranker("normalized.parquet")
scores = r.calculate_scores()
ranked = r.rsd_sort(outname="rsd_scores.parquet")
```

Alternative ranker using Relative Standard Deviation — faster but less
discriminating than PIRS for large datasets.

### `circ.limbr.simulations.simulate` and `circ.pirs.simulations.simulate`

Both modules expose a `simulate()` factory for generating synthetic datasets
suitable for method benchmarking:

```python
from circ.limbr.simulations import simulate   # proteomics simulation (with missing data)
from circ.pirs.simulations import simulate    # expression simulation (const vs rhythmic)

sim = simulate(tpoints=12, nrows=500, nreps=2, pcirc=0.3, rseed=42)
sim.write_output(out_name="sim")
```

The LIMBR simulation additionally supports `generate_pool_map()` for
proteomics pool control files.

## License and attribution

CIRC is released under the BSD 3-Clause License (see `LICENSE`).

It incorporates code from three upstream projects, each with their own license.
Full license texts and attribution details are in `NOTICE`.

| Component | Original authors | License |
|---|---|---|
| `circ.limbr` | Alexander M. Crowell (2017) | BSD 3-Clause |
| `circ.pirs` | Alexander M. Crowell (2017) | BSD 3-Clause |
| `circ.bootjtk` | Alan L. Hutchison (2016); Alexander M. Crowell | MIT |
| `circ/bootjtk/mpfit.py` | More' et al. → C. Markwardt → M. Rivers | — |

### Key publications

- **LIMBR**: Crowell AM et al. "Leveraging information across HeLa cell
  circadian proteomics experiments using a Bayesian multi-study factor
  model." *Molecular & Cellular Proteomics* 2020.
- **BooteJTK**: Hutchison AL, Maienschein-Cline M, Chiang AH et al.
  "Improved statistical methods enable greater sensitivity in rhythm
  detection for genome-wide data." *PLoS Computational Biology* 2015
  11(3): e1004094. doi:10.1371/journal.pcbi.1004094
- **BooteJTK (bootstrap extension)**: Hutchison AL et al. "BooteJTK:
  Improved Rhythm Detection via Bootstrapping." *bioRxiv* 2016.
