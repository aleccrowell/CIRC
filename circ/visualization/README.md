# circ.visualization — Classification and benchmark plots

Classification plots and benchmark evaluation plots for `Classifier` output.
All plot functions accept an optional `ax` keyword for composing multi-panel
figures, and return the `Axes` object.

For interactive (Plotly) versions of these plots see
[`circ.visualization.interactive`](#cirvisualizationinteractive--interactive-plotly-plots).

## Python API

```python
import circ.visualization as viz
import matplotlib.pyplot as plt

# --- Classification plots ---
# Bar chart of gene counts per label
viz.label_distribution(result, ax=ax)

# Scatter of PIRS score vs TauMean with decision boundaries
viz.pirs_vs_tau(result, pirs_percentile=50, tau_threshold=0.5, ax=ax)

# Volcano: PIRS score vs −log₁₀(empirical p-value)
viz.volcano(result, emp_p_threshold=0.05, ax=ax)

# KDE of PIRS scores split by label
viz.pirs_score_distribution(result, ax=ax)

# Scatter of TauMean vs −log₁₀(GammaBH)
viz.tau_pval_scatter(result, ax=ax)

# Polar histogram of estimated phase angles (rhythmic genes only)
viz.phase_wheel(result, labels=("rhythmic", "noisy_rhythmic"), ax=ax)

# Histogram of estimated periods
viz.period_distribution(result, reference_period=24.0, ax=ax)

# Adaptive multi-panel summary figure
fig = viz.classification_summary(result, outpath="summary.png")

# --- Benchmark plots (require simulation ground-truth labels) ---
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

`LABEL_COLORS` maps each label to a consistent hex color for custom plots:

```python
from circ.visualization import LABEL_COLORS
# {'constitutive': '#4878CF', 'rhythmic': '#6ACC65', 'linear': '#D65F5F',
#  'variable': '#B47CC7', 'noisy_rhythmic': '#C4AD66', 'unclassified': '#8C8C8C'}
```

## `circ.visualization.interactive` — Interactive (Plotly) plots

The `interactive` sub-module provides Plotly equivalents of all static plots.
Functions return `plotly.graph_objects.Figure` objects that can be displayed
in Jupyter notebooks or saved with `fig.write_html()`.  The key advantage over
the static versions is that scatter plots carry gene IDs in hover tooltips,
making it easy to identify specific genes of interest.

Requires plotly, available as an optional dependency:

```bash
poetry install --extras interactive
```

```python
import circ.visualization.interactive as iviz

# All static plots have an interactive counterpart
fig = iviz.label_distribution(result)
fig = iviz.pirs_vs_tau(result)          # hover shows gene ID + scores
fig = iviz.volcano(result)              # hover shows gene ID + p-value
fig = iviz.pirs_score_distribution(result)
fig = iviz.tau_pval_scatter(result)
fig = iviz.phase_wheel(result)          # interactive polar bar chart
fig = iviz.period_distribution(result)
fig = iviz.classification_summary(result)   # adaptive multi-panel

# Benchmark plots with threshold hover
fig = iviz.classification_pr(result, true_classes)
fig = iviz.classification_roc(result, true_classes)

# Save to HTML for sharing
fig.write_html("results.html")
```
