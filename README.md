# CIRC — Circadian Integrated Research Core

CIRC is a unified Python toolkit for circadian genomics and proteomics analysis,
integrating four complementary tools:

| Module | Purpose |
|---|---|
| `circ.limbr` | KNN imputation + SVA-based batch effect removal |
| `circ.pirs` | Prediction Interval Ranking Score for constitutive expression |
| `circ.rhythmicity` | BooteJTK + ECHO circadian rhythmicity detection and classification |
| `circ.expression_classification` | Unified classifier combining PIRS, BooteJTK, and ECHO |
| `circ.visualization` | Classification, comparison, and benchmark plots |
| `circ.compare` | Cross-condition and cross-omics comparison |

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

For optional Numba acceleration of BooteJTK (part of `circ.rhythmicity`):

```bash
poetry install --extras fast
```

## Data format

All tools expect expression data with samples as columns named `ZT{time}_{rep}`
(Zeitgeber Time as any non-negative integer, underscore, replicate number).
Short experiments typically use two-digit times; longer time series (> 99 h)
use three or more digits — all are handled identically:

```
#       ZT02_1  ZT02_2  ZT04_1  ZT04_2  ZT06_1  ZT06_2
gene_a  1.23    1.19    0.95    1.01    0.88    0.92
gene_b  ...
```

```
#       ZT100_1  ZT100_2  ZT104_1  ZT104_2  ZT108_1  ZT108_2
gene_a  1.23     1.19     0.95     1.01     0.88     0.92
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

## Gallery

**Classification overview** — label distribution, PIRS score by class, and
mean expression profile per label:

![Classification overview](docs/figures/01_overview.png)

**Cross-omics comparison** — TauMean scatter coloured by rhythmicity status
(maintained rhythmic, gained, lost, non-rhythmic) across proteomics and
RNA-seq layers:

![Cross-omics rhythmicity scatter](docs/figures/33_cross_layer_rhythmicity_scatter.png)

## Examples

Seven self-contained scripts in [`examples/`](examples/) walk through the full
CIRC workflow — from a single classification run to a proteomics pipeline with
batch-effect removal, cross-omics comparison, and visualization.  See the
[examples README](examples/README.md) for the full catalogue.

```bash
# Quickest start: classify simulated data and explore the results
poetry run python examples/01_classify_and_explore.py

# Full proteomics pipeline: impute → batch-correct → classify → benchmark
poetry run python examples/04_proteomics_pipeline.py
```

Pass `--show` to any script to display figures interactively in addition to
saving them.  All figures are written to `./figures/`.

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
result = clf.run_all(slope_pvals=True, n_permutations=1000, n_jobs=4, echo=True)
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
| `classify` | PIRS + BooteJTK (+ ECHO) | Full expression classification pipeline |
| `rhythm` | BooteJTK | Bootstrap JTK rhythm detection |
| `rhythm-calcp` | BooteJTK | Bootstrap JTK + CalcP full pipeline |

For full flag reference see the module READMEs:
[`circ impute` / `circ normalize`](circ/limbr/README.md) ·
[`circ rank`](circ/pirs/README.md) ·
[`circ classify`](circ/expression_classification/README.md) ·
[`circ rhythm` / `circ rhythm-calcp`](circ/rhythmicity/README.md)

## Module API

Full API reference for each module lives in its own README:
[`circ.limbr`](circ/limbr/README.md) ·
[`circ.pirs`](circ/pirs/README.md) ·
[`circ.rhythmicity`](circ/rhythmicity/README.md) ·
[`circ.expression_classification`](circ/expression_classification/README.md) ·
[`circ.visualization`](circ/visualization/README.md)

### `circ.compare` — condition and cross-omics comparison

```python
from circ.compare import compare_conditions, aggregate_to_protein, label_change_table

# Compare two Classifier result DataFrames (same gene IDs, different conditions
# or molecular layers).  Returns per-gene effect sizes and significance tests.
comparison = compare_conditions(result_A, result_B)

# Key output columns:
#   rhythmicity_status  — "maintained_rhythmic" / "gained" / "lost" /
#                          "maintained_nonrhythmic"
#   delta_tau           — TauMean B − A
#   delta_phase         — circular phase shift B − A (h), wrapped to ±12 h
#   tau_padj            — BH-adjusted p-value for rhythmicity change
#                          (requires bootstrap uncertainty columns from BooteJTK)
#   phase_padj          — BH-adjusted p-value for phase shift

# Readable label-transition table (condition A labels → condition B labels)
label_change_table(comparison)

# Cross-omics: collapse peptide-level proteomics results to protein level
# before comparing with a gene-expression result
prot = aggregate_to_protein(peptide_level_result)
comparison = compare_conditions(prot, rna_result)
```

Comparison plots live in `circ.visualization` — see
[`circ/visualization/README.md`](circ/visualization/README.md) for
`rhythmicity_shift_scatter`, `phase_shift_histogram`, `label_transition_heatmap`,
`delta_tau_volcano`, and `comparison_summary`.

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

### `circ.simulations.simulate`

Generates a three-class synthetic time-series (circadian / linear / constitutive)
with optional batch effects and missing data:

```python
from circ.simulations import simulate
# circ.limbr.simulations and circ.pirs.simulations both re-export the same class.

# Minimal expression-style simulation (no noise, no missing data)
sim = simulate(tpoints=12, nrows=500, nreps=2, pcirc=0.3, plin=0.2, rseed=42)
sim.write_output(out_name="sim")          # sim.txt + sim_true_classes.txt

# Proteomics-style simulation (batch effects + missing data)
sim = simulate(
    tpoints=12, nrows=500, nreps=2,
    pcirc=0.3, plin=0.2,
    n_batch_effects=2, pbatch=0.5, effect_size=2.0,
    p_miss=0.3, lam_miss=5,
    rseed=42,
)
sim.generate_pool_map(out_name="pool_map")   # pool_map.parquet
sim.write_proteomics(out_name="sim")         # sim_with_noise.txt + sim_baseline.txt
                                             # + sim_true_classes.txt
```

Key attributes after construction: `sim.classes` (per-row string labels),
`sim.sim` (clean scaled matrix), `sim.sim_miss` (with batch effects and NaN).

## License and attribution

CIRC is released under the BSD 3-Clause License (see `LICENSE`).

It incorporates code from three upstream projects, each with their own license.
Full license texts and attribution details are in `NOTICE`.

| Component | Original authors | License |
|---|---|---|
| `circ.limbr` | Alexander M. Crowell (2017) | BSD 3-Clause |
| `circ.pirs` | Alexander M. Crowell (2017) | BSD 3-Clause |
| `circ.rhythmicity` (BooteJTK) | Alan L. Hutchison (2016) | MIT |
| `circ/rhythmicity/mpfit.py` | More' et al. → C. Markwardt → M. Rivers | — |

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
- **ECHO**: De los Santos H et al. "ECHO: an application for detection
  and analysis of oscillators identifies metabolic regulation on genome-wide
  circadian output." *Bioinformatics* 2019 36(3): 773–781.
  doi:10.1093/bioinformatics/btz617
