# circ.expression_classification — Unified PIRS + BooteJTK classifier

Combines PIRS scores and BooteJTK rhythmicity results to classify every gene
into one of five expression patterns.

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
