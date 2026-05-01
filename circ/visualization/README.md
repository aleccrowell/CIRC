# circ.visualization — Classification and benchmark plots

Classification, comparison, and benchmark plots for `Classifier` output.
All static plot functions accept an optional `ax` keyword for composing
multi-panel figures and return the `Axes` object.

For interactive (Plotly) versions see
[`circ.visualization.interactive`](#cirvisualizationinteractive--interactive-plotly-plots).

## Python API

```python
import circ.visualization as viz
import matplotlib.pyplot as plt
```

### Classification plots

```python
# Bar chart of gene counts per label
viz.label_distribution(result, ax=ax)
# Optional xlim= to fix the x-axis upper limit (useful for side-by-side charts)
viz.label_distribution(result_A, ax=axes[0], xlim=xmax)

# Scatter of PIRS score vs TauMean with decision boundaries
viz.pirs_vs_tau(result, pirs_percentile=50, tau_threshold=0.5, ax=ax)

# Volcano: PIRS score vs −log₁₀(empirical p-value), with quadrant labels
viz.volcano(result, emp_p_threshold=0.05, ax=ax)

# KDE of PIRS scores split by label
viz.pirs_score_distribution(result, ax=ax)

# PIRS score vs −log₁₀(GammaBH p-value)
viz.pirs_pval_scatter(result, ax=ax)

# TauMean vs −log₁₀(GammaBH p-value)
viz.tau_pval_scatter(result, ax=ax)

# Slope significance vs rhythmicity significance
viz.slope_pval_scatter(result, ax=ax)

# Slope p-value vs TauMean scatter
viz.slope_vs_rhythm(result, ax=ax)

# Polar histogram of estimated phase angles (rhythmic genes only)
viz.phase_wheel(result, labels=("rhythmic", "noisy_rhythmic"), ax=ax)

# Histogram of estimated periods
viz.period_distribution(result, reference_period=24.0, ax=ax)

# Phase vs amplitude scatter
viz.phase_amplitude_scatter(result, ax=ax)

# Ranked bar chart of top constitutive gene candidates
viz.top_constitutive_candidates(result, n_top=20, ax=ax)

# Mean expression line plot per label (requires raw expression matrix)
viz.mean_expression_profiles(expression, result, ax=ax)

# Single-gene time-series: replicates as scatter + per-timepoint mean line
# Color is derived from the gene's label; title shows τ, phase, and PIRS
viz.gene_profile(expression, gene_id, result, ax=ax)
viz.gene_profile(expression, gene_id, ax=ax)           # no classifications

# Clustered expression heatmap grouped by label
# Genes are z-scored, ranked by label-specific score, and sorted by
# within-group hierarchical clustering. A color strip encodes the label.
viz.expression_heatmap(expression, result, n_per_label=20, ax=ax)
viz.expression_heatmap(expression, ax=ax)              # no classifications

# How classification counts change as the PIRS threshold sweeps
viz.threshold_sensitivity(result, pirs_percentile=50, ax=ax)

# Adaptive multi-panel summary (panels present depend on available columns)
fig = viz.classification_summary(result, outpath="summary.png")
```

### Comparison plots

Functions for visualizing the output of `circ.compare.compare_conditions()`.

```python
# TauMean scatter coloured by rhythmicity status (maintained / gained / lost)
viz.rhythmicity_shift_scatter(comparison, alpha=0.05, ax=ax)

# Histogram of circular phase differences (genes rhythmic in both conditions)
viz.phase_shift_histogram(comparison, ax=ax)

# Heatmap of label transitions from condition A to condition B
viz.label_transition_heatmap(comparison, ax=ax)

# Volcano: Δ TauMean vs −log₁₀(tau_padj)
# Requires bootstrap uncertainty columns (tau_std, n_boots) in both results
viz.delta_tau_volcano(comparison, alpha=0.05, ax=ax)

# Adaptive multi-panel summary (scatter + heatmap + histogram + volcano)
fig = viz.comparison_summary(comparison, outpath="comparison.png")
```

### Benchmark plots

```python
import pandas as pd
true_classes = pd.read_csv("sim_true_classes.txt", sep="\t", index_col=0)
# Columns: Circadian, Linear, Const (binary 0/1)

# PR curve: how well PIRS score recovers constitutive genes
viz.classification_pr(result, true_classes, ground_truth_col="Const",
                      score_col="pirs_score", ax=ax)

# ROC curves: auto-detects available score × truth pairings
viz.classification_roc(result, true_classes, ax=ax)
# or specify tasks explicitly: (score_col, truth_col, invert_score)
viz.classification_roc(result, true_classes,
                       tasks=[("tau_mean", "Circadian", False),
                               ("slope_pval", "Linear", True)], ax=ax)
```

## Color palettes

`LABEL_COLORS` maps each label to a consistent hex color (Okabe-Ito palette,
safe for the most common forms of color blindness) for use in custom plots:

```python
from circ.visualization import LABEL_COLORS
# {'constitutive': '#0072B2', 'rhythmic': '#009E73', 'linear': '#D55E00',
#  'variable': '#CC79A7', 'noisy_rhythmic': '#E69F00', 'unclassified': '#8C8C8C'}
```

`_STATUS_COLORS` maps each rhythmicity-change category to a consistent color:

```python
from circ.visualization.compare import _STATUS_COLORS
# {'maintained_rhythmic': '#009E73', 'gained': '#D55E00',
#  'lost': '#0072B2', 'maintained_nonrhythmic': '#CCCCCC'}
```

## `circ.visualization.interactive` — Interactive (Plotly) plots

The `interactive` sub-module provides Plotly equivalents of the static plots.
Functions return `plotly.graph_objects.Figure` objects that can be displayed
in Jupyter notebooks or saved with `fig.write_html()`. Scatter plots carry
gene IDs in hover tooltips, making it easy to identify specific genes.

Requires plotly, available as an optional dependency:

```bash
poetry install --extras interactive
```

```python
import circ.visualization.interactive as iviz

# Classification plots — all carry gene IDs in hover tooltips
fig = iviz.label_distribution(result)
fig = iviz.pirs_vs_tau(result)
fig = iviz.volcano(result)
fig = iviz.pirs_score_distribution(result)
fig = iviz.pirs_pval_scatter(result)
fig = iviz.tau_pval_scatter(result)
fig = iviz.slope_pval_scatter(result)
fig = iviz.slope_vs_rhythm(result)
fig = iviz.phase_wheel(result)
fig = iviz.period_distribution(result)
fig = iviz.phase_amplitude_scatter(result)
fig = iviz.top_constitutive_candidates(result)
fig = iviz.expression_heatmap(expression, result)  # hover shows gene + z-score
fig = iviz.classification_summary(result)          # adaptive multi-panel

# Comparison plots
fig = iviz.rhythmicity_shift_scatter(comparison)   # hover shows gene ID + Δτ
fig = iviz.delta_tau_volcano(comparison)           # hover shows gene ID + padj

# Benchmark plots
fig = iviz.classification_pr(result, true_classes)
fig = iviz.classification_roc(result, true_classes)

# Save to HTML for sharing
fig.write_html("results.html")
```
