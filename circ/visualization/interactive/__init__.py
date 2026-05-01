"""Interactive (Plotly) visualizations for CIRC expression analysis.

All functions return ``plotly.graph_objects.Figure`` objects that can be
displayed in Jupyter notebooks or saved with ``fig.write_html()``.

Requires plotly, available as an optional dependency::

    poetry install --extras interactive

Classification plots
--------------------
.. autosummary::
   label_distribution
   pirs_vs_tau
   volcano
   pirs_score_distribution
   tau_pval_scatter
   phase_wheel
   period_distribution
   classification_summary

Benchmark / evaluation plots
-----------------------------
.. autosummary::
   classification_pr
   classification_roc
"""

try:
    import plotly.graph_objects  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "circ.visualization.interactive requires plotly. "
        "Install it with: poetry install --extras interactive"
    ) from exc

from circ.visualization.interactive.classify import (
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
    classification_summary,
)
from circ.visualization.interactive.benchmarks import (
    classification_pr,
    classification_roc,
)

__all__ = [
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
    "classification_summary",
    "classification_pr",
    "classification_roc",
]
