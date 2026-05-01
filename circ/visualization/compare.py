"""Visualizations for between-condition comparison results from circ.compare."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from circ.visualization.classify import LABEL_COLORS, _ax

# Colors and draw order for rhythmicity change categories
_STATUS_COLORS = {
    "maintained_rhythmic": "#6ACC65",
    "gained": "#D65F5F",
    "lost": "#4878CF",
    "maintained_nonrhythmic": "#CCCCCC",
}
_STATUS_ORDER = ["maintained_rhythmic", "gained", "lost", "maintained_nonrhythmic"]


def rhythmicity_shift_scatter(
    comparison,
    alpha=0.05,
    ax=None,
    title="Rhythmicity shift: TauMean per condition",
):
    """Scatter of tau_mean_A vs tau_mean_B, colored by rhythmicity status.

    Each point is a shared gene.  Points above the diagonal increased
    rhythmicity in condition B; below the diagonal lost it.  When
    ``tau_padj`` is present, genes with a significant adjusted p-value
    are drawn with larger, opaque markers; the rest are small and faint.

    Parameters
    ----------
    comparison : pd.DataFrame
        Output of :func:`~circ.compare.compare_conditions`.
    alpha : float
        FDR threshold for highlighting significant tau changes (default 0.05).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ax(ax)
    has_padj = "tau_padj" in comparison.columns

    for status in _STATUS_ORDER:
        sub = comparison[comparison["rhythmicity_status"] == status]
        if sub.empty:
            continue
        color = _STATUS_COLORS[status]
        if has_padj:
            sig = sub["tau_padj"] < alpha
            if sig.any():
                ax.scatter(
                    sub.loc[sig, "tau_mean_A"],
                    sub.loc[sig, "tau_mean_B"],
                    color=color,
                    s=28,
                    alpha=0.85,
                    zorder=4,
                    edgecolors="white",
                    linewidths=0.5,
                    label=f"{status} (FDR<{alpha})",
                )
            if (~sig).any():
                ax.scatter(
                    sub.loc[~sig, "tau_mean_A"],
                    sub.loc[~sig, "tau_mean_B"],
                    color=color,
                    s=10,
                    alpha=0.35,
                    zorder=3,
                    label=status,
                )
        else:
            ax.scatter(
                sub["tau_mean_A"],
                sub["tau_mean_B"],
                color=color,
                s=10,
                alpha=0.55,
                zorder=3,
                label=status,
            )

    lim = max(comparison["tau_mean_A"].max(), comparison["tau_mean_B"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#999999", ls="--", lw=0.9, alpha=0.6)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("TauMean — condition A")
    ax.set_ylabel("TauMean — condition B")
    ax.set_title(title)
    patches = [
        mpatches.Patch(color=_STATUS_COLORS[s], label=s)
        for s in _STATUS_ORDER
        if s in comparison["rhythmicity_status"].values
    ]
    ax.legend(handles=patches, frameon=False, fontsize=9)
    sns.despine(ax=ax)
    return ax


def phase_shift_histogram(
    comparison,
    ax=None,
    title="Phase shift distribution (rhythmic in both conditions)",
):
    """Histogram of circular phase differences for genes rhythmic in both conditions.

    ``delta_phase`` is the signed difference phase_B − phase_A, wrapped
    to ±12 h.  A vertical dashed line at 0 marks no change.  Only genes
    with a valid ``delta_phase`` (i.e. rhythmic in both conditions) are
    shown.

    Parameters
    ----------
    comparison : pd.DataFrame
        Output of :func:`~circ.compare.compare_conditions`.
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ax(ax)

    if "delta_phase" not in comparison.columns:
        ax.text(
            0.5,
            0.5,
            "No phase data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#888888",
        )
        ax.set_title(title)
        return ax

    dp = comparison["delta_phase"].dropna()
    if dp.empty:
        ax.text(
            0.5,
            0.5,
            "No genes rhythmic\nin both conditions",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#888888",
        )
        ax.set_title(title)
        sns.despine(ax=ax)
        return ax

    bins = np.arange(-12, 13, 2)
    ax.hist(
        dp, bins=bins, color=LABEL_COLORS["rhythmic"], alpha=0.75, edgecolor="white"
    )
    ax.axvline(0, color="#333333", ls="--", lw=0.9, alpha=0.8, label="no shift")
    ax.set_xlabel("Phase shift B − A (h)")
    ax.set_ylabel("Gene count")
    ax.set_xlim(-12, 12)
    ax.set_xticks(range(-12, 13, 4))
    ax.set_xticklabels([f"{h:+d}" if h != 0 else "0" for h in range(-12, 13, 4)])
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    sns.despine(ax=ax)
    return ax


def label_transition_heatmap(
    comparison,
    ax=None,
    title="Label transitions A → B",
):
    """Heatmap of classification label changes between conditions.

    Each cell shows the number of genes that moved from label A (row)
    to label B (column).  Diagonal cells are genes whose label did not
    change.

    Parameters
    ----------
    comparison : pd.DataFrame
        Output of :func:`~circ.compare.compare_conditions`.
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    from circ.compare import label_change_table

    ax = _ax(ax)
    ct = label_change_table(comparison)
    if ct.empty:
        ax.set_title(title)
        return ax

    sns.heatmap(
        ct,
        ax=ax,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        annot_kws={"size": 9},
    )
    ax.set_xlabel("Label in condition B")
    ax.set_ylabel("Label in condition A")
    ax.set_title(title)
    return ax


def delta_tau_volcano(
    comparison,
    alpha=0.05,
    ax=None,
    title="Rhythmicity change: Δ TauMean vs significance",
):
    """Volcano plot of Δ TauMean vs −log₁₀(tau_padj).

    Each point is a shared gene.  The x-axis shows TauMean B − A
    (positive = more rhythmic in B); the y-axis shows significance.
    A horizontal dashed line marks the FDR threshold at *alpha*.

    Requires ``tau_padj`` (produced when ``tau_std`` and ``n_boots`` are
    present in both result DataFrames — i.e. when BooteJTK was run and
    the Classifier added uncertainty columns to its output).

    Parameters
    ----------
    comparison : pd.DataFrame
        Output of :func:`~circ.compare.compare_conditions`.
    alpha : float
        FDR threshold drawn as a horizontal line (default 0.05).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if "tau_padj" not in comparison.columns:
        raise ValueError(
            "'tau_padj' column missing — re-run compare_conditions() with "
            "result DataFrames that include tau_std and n_boots (produced "
            "automatically when Classifier.classify() is run after run_bootjtk())."
        )
    ax = _ax(ax)
    df = comparison.dropna(subset=["delta_tau", "tau_padj"])
    df = df.assign(neg_log_p=-np.log10(df["tau_padj"].clip(lower=1e-300)))

    for status in _STATUS_ORDER:
        sub = df[df["rhythmicity_status"] == status]
        if sub.empty:
            continue
        ax.scatter(
            sub["delta_tau"],
            sub["neg_log_p"],
            color=_STATUS_COLORS[status],
            s=14,
            alpha=0.65,
            rasterized=len(df) > 500,
        )

    ax.axhline(
        -np.log10(alpha),
        color="#333333",
        ls="--",
        lw=0.9,
        alpha=0.7,
        label=f"FDR = {alpha}",
    )
    ax.axvline(0, color="#999999", ls=":", lw=0.8, alpha=0.5)

    ax.set_xlabel("Δ TauMean (B − A)")
    ax.set_ylabel("−log₁₀(tau_padj)")
    ax.set_title(title)
    patches = [
        mpatches.Patch(color=_STATUS_COLORS[s], label=s)
        for s in _STATUS_ORDER
        if s in df["rhythmicity_status"].values
    ]
    ax.legend(handles=patches, frameon=False, fontsize=9)
    sns.despine(ax=ax)
    return ax


def comparison_summary(comparison, outpath=None):
    """Multi-panel summary figure for a condition comparison.

    Panels shown (based on available columns):

    * Always: rhythmicity shift scatter, label transition heatmap
    * Requires ``delta_phase``: phase shift histogram
    * Requires ``tau_padj``: delta tau volcano

    Parameters
    ----------
    comparison : pd.DataFrame
        Output of :func:`~circ.compare.compare_conditions`.
    outpath : str or None
        If provided, save to this path (dpi = 150).

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels = [
        lambda ax: rhythmicity_shift_scatter(comparison, ax=ax),
        lambda ax: label_transition_heatmap(comparison, ax=ax),
    ]
    if "delta_phase" in comparison.columns:
        panels.append(lambda ax: phase_shift_histogram(comparison, ax=ax))
    if "tau_padj" in comparison.columns:
        panels.append(lambda ax: delta_tau_volcano(comparison, ax=ax))

    n = len(panels)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    for ax, fn in zip(axes_flat, panels):
        fn(ax)
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout(pad=1.5, h_pad=2.0, w_pad=1.5)
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
    return fig
