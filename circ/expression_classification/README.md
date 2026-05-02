# circ.expression_classification — Unified PIRS + BooteJTK + ECHO classifier

Combines PIRS constitutiveness scores, BooteJTK rhythmicity results, and
optionally ECHO amplitude-aware fitting to classify every gene into one of
five expression patterns.

## Expression labels

| Label | Description |
|---|---|
| `constitutive` | Stable expression, no significant slope, not rhythmic |
| `rhythmic` | Stable-to-moderate expression with a strong circadian rhythm |
| `linear` | Significant linear slope, not rhythmic |
| `variable` | High PIRS score (non-constitutive), no slope, not rhythmic |
| `noisy_rhythmic` | High PIRS score with a detectable rhythmic signal |

`linear` is only emitted when slope p-values have been computed (see below).
Without slope p-values the original four-label scheme is used.

## Python API

### `circ.expression_classification.classify.Classifier`

```python
from circ.expression_classification.classify import Classifier

# Accepts a file path (TSV or Parquet) or a DataFrame
clf = Classifier("normalized.parquet", anova=False, reps=2, size=50, workers=1)
```

#### Step-by-step

```python
# 1. PIRS constitutiveness scores
clf.run_pirs(pvals=True, slope_pvals=True, n_permutations=1000, n_jobs=4)

# 2. BooteJTK rhythmicity detection
clf.run_bootjtk()

# 3. (Optional) ECHO amplitude-aware fitting
clf.run_echo(workers=4)

# 4. Classify — pass echo=True to include echo_amplitude_class in output
result = clf.classify(
    pirs_percentile=50,       # genes at/below this PIRS percentile are "stable"
    slope_pval_threshold=0.05,
    tau_threshold=0.5,
    emp_p_threshold=0.05,
    echo=True,
    echo_p_threshold=0.05,    # amplitude class masked for genes above this BH p-value
)
```

#### Single-call convenience

```python
result = clf.run_all(
    slope_pvals=True,
    n_permutations=1000,
    n_jobs=4,
    echo=True,                # also runs run_echo() and includes ECHO columns
)
```

`result` is a DataFrame indexed by gene ID.  Core columns:

| Column | Description |
|---|---|
| `pirs_score` | Raw PIRS constitutiveness score |
| `slope_pval` / `slope_pval_bh` | Slope p-values (when `slope_pvals=True`) |
| `tau_mean` | BooteJTK mean Kendall's τ across bootstraps |
| `emp_p` | BH-corrected BooteJTK empirical p-value |
| `period_mean` | Estimated period (hours) |
| `phase_mean` | Estimated phase (hours) |
| `label` | Expression class (see table above) |

Additional ECHO columns (present when `echo=True`):

| Column | Description |
|---|---|
| `echo_amplitude_class` | `'damped'`, `'harmonic'`, or `'forced'`; `None` if not significant |
| `echo_gamma` | Amplitude change coefficient |
| `echo_period` | ECHO-fitted period (hours) |
| `echo_phase` | ECHO-fitted phase (hours) |
| `echo_tau` | Kendall's τ goodness-of-fit |
| `echo_p_bh` | BH-corrected ECHO p-value |

## CLI

### `circ classify`

```bash
circ classify -f normalized.parquet -o classified.txt \
    --reps 2 --size 50 --workers 4 \
    --pirs-percentile 50 \
    --tau-threshold 0.5 \
    --emp-p-threshold 0.05

# With ECHO amplitude-aware classification
circ classify -f normalized.parquet -o classified.txt \
    --reps 2 --size 50 --workers 4 \
    --echo --echo-p-threshold 0.05
```

| Flag | Default | Description |
|---|---|---|
| `-f FILE` | required | Input expression file (TSV or Parquet) |
| `-o FILE` | required | Output file path |
| `--anova` | off | ANOVA-filter differentially expressed genes before PIRS |
| `--pirs-percentile` | 50 | PIRS percentile cutoff for "stable" genes |
| `--tau-threshold` | 0.5 | Minimum TauMean for rhythmicity call |
| `--emp-p-threshold` | 0.05 | Maximum FDR-corrected p-value for rhythmicity |
| `-r / --reps` | 2 | Replicates per timepoint (BooteJTK) |
| `-z / --size` | 50 | Bootstrap iterations per gene (BooteJTK) |
| `-j / --workers` | 1 | Parallel worker processes |
| `--limma` | off | Apply Limma/Vash variance shrinkage before BooteJTK |
| `--echo` | off | Run ECHO amplitude-aware fitting |
| `--echo-p-threshold` | 0.05 | BH p-value cutoff for ECHO amplitude classification |
