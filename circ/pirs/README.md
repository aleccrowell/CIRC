# circ.pirs — Prediction Interval Ranking Score

PIRS ranks expression profiles by their constitutiveness: how tightly a gene's
expression hugs its mean over time.  Lower score = more constitutive.

The score for each gene is:

```
max over fine time grid of max(|PI_upper(t) − mean_expr|, |PI_lower(t) − mean_expr|)
────────────────────────────────────────────────────────────────────────────────────
                              |mean_expr|
```

A flat, low-noise gene has tight prediction interval bounds near mean expression
everywhere → score near 0.  A trending or noisy gene has bounds that deviate far
from mean expression → large score.

When `anova=True` (default), rhythmic profiles are removed by one-way ANOVA
before scoring so that constitutive candidates are not contaminated by
circadian genes.

## CLI

### `circ rank`

```
circ rank -f INPUT -o OUTPUT [--no-anova]
```

| Flag | Default | Description |
|---|---|---|
| `-f/--filename` | required | Normalized input file (# index + ZT columns) |
| `-o/--output` | required | Output scores file |
| `--no-anova` | off | Skip ANOVA pre-filtering of rhythmic profiles |

## Python API

### `circ.pirs.rank.ranker`

```python
from circ.pirs.rank import ranker

# Accepts a file path (TSV or Parquet) or a DataFrame
r = ranker("normalized.parquet", anova=True)
r.get_tpoints()
scores = r.calculate_scores()                 # returns DataFrame sorted ascending
ranked = r.pirs_sort(outname="scores.parquet")  # writes file; returns sorted data
```

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
