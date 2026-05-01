"""Visualization utilities for CIRC expression analysis results.

Classification plots
--------------------
.. autosummary::
   label_distribution
   pirs_vs_tau
   volcano
   pirs_score_distribution
   tau_pval_scatter
   pirs_pval_scatter
   slope_pval_scatter
   slope_vs_rhythm
   phase_wheel
   period_distribution
   classification_summary

Condition comparison plots
--------------------------
.. autosummary::
   rhythmicity_shift_scatter
   phase_shift_histogram
   label_transition_heatmap
   delta_tau_volcano
   comparison_summary

Benchmark / evaluation plots
-----------------------------
.. autosummary::
   pr_curve
   roc_curve_plot
   roc_auc
   classification_pr
   classification_roc

The ``LABEL_COLORS`` dict maps each expression label to a consistent hex
color and can be imported for use in custom plots.
"""

from circ.visualization.classify import (
    LABEL_COLORS,
    label_distribution,
    pirs_vs_tau,
    volcano,
    pirs_score_distribution,
    tau_pval_scatter,
    pirs_pval_scatter,
    slope_pval_scatter,
    slope_vs_rhythm,
    phase_wheel,
    period_distribution,
    phase_amplitude_scatter,
    top_constitutive_candidates,
    mean_expression_profiles,
    gene_profile,
    expression_heatmap,
    threshold_sensitivity,
    classification_summary,
)
from circ.visualization.compare import (
    rhythmicity_shift_scatter,
    phase_shift_histogram,
    label_transition_heatmap,
    delta_tau_volcano,
    comparison_summary,
)
from circ.evaluation import roc_auc
from circ.visualization.benchmarks import (
    pr_curve,
    roc_curve_plot,
    classification_pr,
    classification_roc,
)

__all__ = [
    # classify
    "LABEL_COLORS",
    "label_distribution",
    "pirs_vs_tau",
    "volcano",
    "pirs_score_distribution",
    "tau_pval_scatter",
    "pirs_pval_scatter",
    "slope_pval_scatter",
    "slope_vs_rhythm",
    "phase_wheel",
    "period_distribution",
    "phase_amplitude_scatter",
    "top_constitutive_candidates",
    "mean_expression_profiles",
    "gene_profile",
    "expression_heatmap",
    "threshold_sensitivity",
    "classification_summary",
    # compare
    "rhythmicity_shift_scatter",
    "phase_shift_histogram",
    "label_transition_heatmap",
    "delta_tau_volcano",
    "comparison_summary",
    # benchmarks
    "pr_curve",
    "roc_curve_plot",
    "roc_auc",
    "classification_pr",
    "classification_roc",
]
