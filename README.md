# CIRC — Circadian Integrated Research Core

CIRC is a unified Python toolkit for circadian genomics and proteomics analysis,
integrating three complementary tools:

| Module | Purpose |
|---|---|
| `circ.limbr` | KNN imputation + SVA-based batch effect removal |
| `circ.pirs` | Prediction Interval Ranking Score for constitutive expression |
| `circ.bootjtk` | Bootstrap empirical JTK circadian rhythm detection |

A single `circ` CLI entry point wraps all three, and all modules share the same
`ZT{HH}_{rep}` column format so outputs chain directly between steps.

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

All three tools expect tab-separated files with samples as columns named
`ZT{HH}_{rep}` (zero-padded two-digit Zeitgeber Time, underscore, replicate
number), for example:

```
#       ZT02_1  ZT02_2  ZT04_1  ZT04_2  ZT06_1  ZT06_2
gene_a  1.23    1.19    0.95    1.01    0.88    0.92
gene_b  ...
```

For proteomics data the first two columns are `Peptide` and `Protein` instead
of the single `#` index column.

## Full pipeline

A typical proteomics circadian experiment:

```python
from circ.limbr import simulations, imputation, batch_fx
from circ.pirs import rank as pirs_rank

# 1. Simulate (or start from real data)
sim = simulations.simulate(tpoints=12, nrows=500, nreps=2, rseed=42)
sim.generate_pool_map(out_name="pool_map")
sim.write_output(out_name="sim")

# 2. Impute missing values
imp = imputation.imputable("sim_with_noise.txt", missingness=0.3, neighbors=10)
imp.impute_data("imputed.txt")

# 3. Remove batch effects
sva_obj = batch_fx.sva("imputed.txt", design="c", data_type="p",
                        pool="pool_map.parquet")
sva_obj.preprocess_default()
sva_obj.perm_test(nperm=200)
sva_obj.output_default("normalized.txt")

# 4. Rank by constitutiveness
ranker = pirs_rank.ranker("normalized.txt", anova=True)
ranked = ranker.pirs_sort(outname="ranked_scores.txt")

# 5. Detect circadian rhythms (via CLI or directly)
# circ rhythm-calcp -f normalized.txt -x output_prefix -r 2 -z 10
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

### `circ.limbr.imputation.imputable`

```python
from circ.limbr.imputation import imputable

obj = imputable(filename, missingness=0.3, neighbors=10)
obj.impute_data(output_filename)
```

Deduplicates peptides, drops rows exceeding the missingness threshold, and
imputes remaining missing values using K-nearest neighbours.

### `circ.limbr.batch_fx.sva`

```python
from circ.limbr.batch_fx import sva

obj = sva(filename, design="c", data_type="p", pool="pool_map.parquet")
obj.preprocess_default()
obj.perm_test(nperm=200, npr=1)
obj.output_default(output_filename)
```

Applies pool normalization (proteomics), quantile normalization, and SVA-based
identification and removal of latent batch effects.

`design` options: `"c"` (circadian), `"t"` (timecourse), `"b"` (blocked).
`data_type` options: `"p"` (proteomics — Peptide/Protein columns), `"r"` (RNAseq — # column).

### `circ.pirs.rank.ranker`

```python
from circ.pirs.rank import ranker

r = ranker(filename, anova=True)
r.get_tpoints()
scores = r.calculate_scores()       # returns DataFrame sorted ascending by score
ranked = r.pirs_sort(outname=None)  # returns DataFrame; writes file if outname given
```

Scores each expression profile by how well its variance is explained by a
prediction interval model — lower score = more constitutive.
When `anova=True` (default), rhythmic profiles are removed by one-way ANOVA
before ranking so that constitutive candidates are not contaminated by
circadian genes.

### `circ.pirs.rank.rsd_ranker`

```python
from circ.pirs.rank import rsd_ranker

r = rsd_ranker(filename)
scores = r.calculate_scores()
ranked = r.rsd_sort(outname=None)
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
