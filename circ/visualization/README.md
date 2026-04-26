# circ.visualization — Classification and benchmark plots

Classification plots and benchmark evaluation plots for `Classifier` output.
All plot functions accept an optional `ax` keyword for composing multi-panel
figures, and return the `Axes` object.

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
