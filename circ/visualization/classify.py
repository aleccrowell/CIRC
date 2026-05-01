"""Visualizations for expression classification results from Classifier.classify()."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Consistent label colors used across every plot in this module
LABEL_COLORS = {
    "constitutive": "#4878CF",
    "rhythmic": "#6ACC65",
    "linear": "#D65F5F",
    "variable": "#B47CC7",
    "noisy_rhythmic": "#C4AD66",
    "unclassified": "#8C8C8C",
}

_LABEL_ORDER = [
    "constitutive",
    "rhythmic",
    "linear",
    "variable",
    "noisy_rhythmic",
    "unclassified",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ax(ax):
    return ax if ax is not None else plt.subplots()[1]


def _safe_neglog10(series, floor=1e-300):
    return -np.log10(np.maximum(series, floor))


def _has(df, col):
    return col in df.columns and df[col].notna().any()


def _scatter_by_label(df, xcol, ycol, ax, size=18, alpha=0.65):
    for lbl in _LABEL_ORDER:
        sub = df[df["label"] == lbl] if "label" in df.columns else df
        if not sub.empty:
            ax.scatter(
                sub[xcol],
                sub[ycol],
                color=LABEL_COLORS.get(lbl, "#8C8C8C"),
                s=size,
                alpha=alpha,
                label=lbl,
                rasterized=len(df) > 1000,
            )


def _label_legend(present, ax):
    patches = [
        mpatches.Patch(color=LABEL_COLORS[l], label=l)
        for l in _LABEL_ORDER
        if l in present
    ]
    if patches:
        ax.legend(handles=patches, loc="best", frameon=False, fontsize=9)


def _zt_timepoints(expression):
    """Return (zt_cols, timepoints_array, unique_tp_array) for an expression DataFrame."""
    cols = [c for c in expression.columns if c.startswith("ZT") or c.startswith("CT")]
    if not cols:
        raise ValueError("No ZT/CT columns found in expression DataFrame.")
    tp = np.array(
        [int(c.replace("ZT", "").replace("CT", "").split("_")[0]) for c in cols]
    )
    return cols, tp, np.sort(np.unique(tp))


def _clip_axes_to_data(ax, x_series, y_series, x_pct=(1, 99), y_pct=(0, 99)):
    """Set axis limits to data percentiles so outliers don't compress the view."""
    x = x_series.dropna()
    y = y_series.dropna()
    if len(x) >= 2:
        xlo, xhi = np.percentile(x, x_pct)
        margin = max((xhi - xlo) * 0.05, 1e-6)
        ax.set_xlim(xlo - margin, xhi + margin)
    if len(y) >= 2:
        ylo, yhi = np.percentile(y, y_pct)
        margin = max((yhi - ylo) * 0.05, 1e-6)
        ax.set_ylim(max(0.0, ylo - margin), yhi + margin)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------


def label_distribution(
    classifications, ax=None, title="Expression label counts", xlim=None
):
    """Horizontal bar chart of gene counts per expression label.

    Parameters
    ----------
    classifications : pd.DataFrame
        Output of ``Classifier.classify()``.
    ax : matplotlib.axes.Axes, optional
    title : str, optional
    xlim : float or None
        If provided, fix the x-axis upper limit to this value.  Useful when
        placing two side-by-side distribution charts on a shared scale::

            xmax = max(result_A["label"].value_counts().max(),
                       result_B["label"].value_counts().max()) * 1.15
            viz.label_distribution(result_A, ax=axes[0], xlim=xmax)
            viz.label_distribution(result_B, ax=axes[1], xlim=xmax)

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ax(ax)
    present = [l for l in _LABEL_ORDER if l in classifications["label"].values]
    counts = (
        classifications["label"].value_counts().reindex(present).fillna(0).astype(int)
    )
    colors = [LABEL_COLORS[l] for l in counts.index]
    bars = ax.barh(counts.index, counts.values, color=colors)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_xlabel("Gene count")
    ax.set_title(title)
    ax.invert_yaxis()
    if xlim is not None:
        ax.set_xlim(0, xlim)
    sns.despine(ax=ax, left=True)
    return ax


def pirs_vs_tau(
    classifications,
    pirs_percentile=50,
    tau_threshold=0.5,
    ax=None,
    title="PIRS score vs rhythmicity (TauMean)",
):
    """Scatter of PIRS constitutiveness score vs BooteJTK TauMean.

    Decision boundaries are drawn at *pirs_percentile* (vertical dashed line)
    and *tau_threshold* (horizontal dotted line), showing the four
    classification quadrants.

    Parameters
    ----------
    classifications : pd.DataFrame
    pirs_percentile : float
        Percentile threshold used for the stable/unstable split (default 50).
    tau_threshold : float
        TauMean threshold for rhythmicity calls (default 0.5).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ax(ax)
    df = classifications.dropna(subset=["pirs_score", "tau_mean"])
    pirs_cut = np.percentile(df["pirs_score"], pirs_percentile)

    _scatter_by_label(df, "pirs_score", "tau_mean", ax)

    ax.axvline(
        pirs_cut,
        color="#333333",
        ls="--",
        lw=0.9,
        alpha=0.7,
        label=f"PIRS p{int(pirs_percentile)}",
    )
    ax.axhline(
        tau_threshold,
        color="#333333",
        ls=":",
        lw=0.9,
        alpha=0.7,
        label=f"τ = {tau_threshold}",
    )
    # Clip x to 95th percentile to match pirs_score_distribution and avoid
    # high-PIRS outliers compressing the constitutive cluster at the left edge
    _clip_axes_to_data(ax, df["pirs_score"], df["tau_mean"], x_pct=(0, 95))
    ax.set_xlabel("PIRS score")
    ax.set_ylabel("TauMean")
    ax.set_title(title)
    _label_legend(df["label"].unique() if "label" in df.columns else [], ax)
    sns.despine(ax=ax)
    return ax


def volcano(
    classifications,
    emp_p_threshold=0.05,
    pirs_percentile=50,
    ax=None,
    title="PIRS score vs rhythmicity significance",
):
    """Scatter of PIRS score vs −log₁₀(emp_p).

    Combines the constitutiveness axis (PIRS score) with the rhythmicity
    significance axis (GammaBH empirical p-value), revealing the four
    quadrants that underlie the classification labels.

    Requires ``emp_p`` column (present when ``run_bootjtk()`` has been called).

    Parameters
    ----------
    classifications : pd.DataFrame
    emp_p_threshold : float
        FDR significance threshold drawn as a horizontal line (default 0.05).
    pirs_percentile : float
        PIRS percentile threshold drawn as a vertical dashed line (default 50).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if not _has(classifications, "emp_p"):
        raise ValueError("'emp_p' column is required. Call run_bootjtk() first.")
    ax = _ax(ax)
    df = classifications.dropna(subset=["pirs_score", "emp_p"])
    df = df.assign(neg_log_emp_p=_safe_neglog10(df["emp_p"]))
    pirs_cut = np.percentile(df["pirs_score"], pirs_percentile)

    _scatter_by_label(df, "pirs_score", "neg_log_emp_p", ax)

    ax.axhline(
        -np.log10(emp_p_threshold),
        color="#333333",
        ls="--",
        lw=0.9,
        alpha=0.7,
        label=f"FDR = {emp_p_threshold}",
    )
    ax.axvline(
        pirs_cut,
        color="#333333",
        ls=":",
        lw=0.9,
        alpha=0.7,
        label=f"PIRS p{int(pirs_percentile)}",
    )
    _clip_axes_to_data(ax, df["pirs_score"], df["neg_log_emp_p"])
    _qstyle = dict(transform=ax.transAxes, fontsize=7, color="#AAAAAA", style="italic")
    ax.text(0.02, 0.04, "constitutive", va="bottom", ha="left", **_qstyle)
    ax.text(0.98, 0.04, "variable", va="bottom", ha="right", **_qstyle)
    ax.text(0.02, 0.96, "noisy_rhythmic", va="top", ha="left", **_qstyle)
    ax.text(0.98, 0.96, "rhythmic", va="top", ha="right", **_qstyle)
    ax.set_xlabel("PIRS score")
    ax.set_ylabel("−log₁₀(GammaBH)")
    ax.set_title(title)
    _label_legend(df["label"].unique() if "label" in df.columns else [], ax)
    sns.despine(ax=ax)
    return ax


def pirs_score_distribution(
    classifications,
    pirs_percentile=50,
    ax=None,
    title="PIRS score distribution by label",
):
    """KDE of PIRS scores overlaid per expression label.

    A vertical dashed line marks the constitutiveness cutoff at
    *pirs_percentile*.

    Parameters
    ----------
    classifications : pd.DataFrame
    pirs_percentile : float
        Percentile cutoff shown as a reference line (default 50).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ax(ax)
    all_scores = classifications["pirs_score"].dropna()
    pirs_cut = np.percentile(all_scores, pirs_percentile)
    for lbl in _LABEL_ORDER:
        sub = classifications[classifications["label"] == lbl]["pirs_score"].dropna()
        if len(sub) >= 3:
            sns.kdeplot(
                sub,
                ax=ax,
                color=LABEL_COLORS[lbl],
                label=lbl,
                fill=True,
                alpha=0.3,
                linewidth=1.2,
            )
    ax.axvline(
        pirs_cut,
        color="#333333",
        ls="--",
        lw=0.9,
        alpha=0.8,
        label=f"p{int(pirs_percentile)} cut",
    )
    # Clip x-axis: always start at 0 and end at the 95th percentile so the
    # constitutive spike and the label separation are both visible.
    hi = np.percentile(all_scores, 95)
    margin = max(hi * 0.05, 0.05)
    ax.set_xlim(0, hi + margin)
    ax.set_xlabel("PIRS score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    sns.despine(ax=ax)
    return ax


def tau_pval_scatter(
    classifications,
    tau_threshold=0.5,
    emp_p_threshold=0.05,
    ax=None,
    title="Rhythmicity: TauMean vs GammaBH significance",
):
    """Scatter of TauMean vs −log₁₀(emp_p), showing the BooteJTK decision space.

    Threshold lines illustrate where the rhythmicity call is made.

    Requires ``emp_p`` column.

    Parameters
    ----------
    classifications : pd.DataFrame
    tau_threshold : float
        TauMean threshold (default 0.5).
    emp_p_threshold : float
        Significance threshold (default 0.05).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if not _has(classifications, "emp_p"):
        raise ValueError("'emp_p' column is required. Call run_bootjtk() first.")
    ax = _ax(ax)
    df = classifications.dropna(subset=["tau_mean", "emp_p"])
    df = df.assign(neg_log_emp_p=_safe_neglog10(df["emp_p"]))

    _scatter_by_label(df, "tau_mean", "neg_log_emp_p", ax)

    ax.axvline(
        tau_threshold,
        color="#333333",
        ls="--",
        lw=0.9,
        alpha=0.7,
        label=f"τ = {tau_threshold}",
    )
    ax.axhline(
        -np.log10(emp_p_threshold),
        color="#333333",
        ls=":",
        lw=0.9,
        alpha=0.7,
        label=f"FDR = {emp_p_threshold}",
    )
    # Always start tau from 0 so the threshold line is never at the left edge
    _clip_axes_to_data(ax, df["tau_mean"], df["neg_log_emp_p"], x_pct=(0, 99))
    ax.set_xlabel("TauMean")
    ax.set_ylabel("−log₁₀(GammaBH)")
    ax.set_title(title)
    _label_legend(df["label"].unique() if "label" in df.columns else [], ax)
    sns.despine(ax=ax)
    return ax


def pirs_pval_scatter(
    classifications,
    pval_threshold=0.05,
    ax=None,
    title="PIRS score vs temporal structure significance",
):
    """Scatter of PIRS score vs −log₁₀(pval_bh).

    Combines the raw PIRS score with its permutation p-value.  Genes with both
    a high PIRS score **and** a significant pval have statistically confirmed
    temporal structure, distinguishing them from genes with high score due to
    noise alone.

    Requires ``pval`` column (from ``run_pirs(pvals=True)``).

    Parameters
    ----------
    classifications : pd.DataFrame
    pval_threshold : float
        BH-corrected p-value threshold drawn as a horizontal line (default 0.05).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    p_col = "pval_bh" if _has(classifications, "pval_bh") else "pval"
    if not _has(classifications, p_col):
        raise ValueError("'pval' column is required. Call run_pirs(pvals=True) first.")
    ax = _ax(ax)
    df = classifications.dropna(subset=["pirs_score", p_col])
    df = df.assign(neg_log_p=_safe_neglog10(df[p_col]))

    _scatter_by_label(df, "pirs_score", "neg_log_p", ax)

    ax.axhline(
        -np.log10(pval_threshold),
        color="#333333",
        ls="--",
        lw=0.9,
        alpha=0.7,
        label=f"α = {pval_threshold}",
    )
    lbl_used = "pval_bh" if p_col == "pval_bh" else "pval"
    _clip_axes_to_data(ax, df["pirs_score"], df["neg_log_p"])
    ax.set_xlabel("PIRS score")
    ax.set_ylabel(f"−log₁₀({lbl_used})")
    ax.set_title(title)
    _label_legend(df["label"].unique() if "label" in df.columns else [], ax)
    sns.despine(ax=ax)
    return ax


def slope_pval_scatter(
    classifications,
    slope_pval_threshold=0.05,
    pirs_percentile=50,
    ax=None,
    title="PIRS score vs linear slope significance",
):
    """Scatter of PIRS score vs −log₁₀(slope_pval_bh).

    Reveals which genes have a statistically significant linear trend in
    expression.  Linear genes should cluster at top-left (low PIRS score but
    significant slope), while constitutive genes sit at the bottom-left.

    Requires ``slope_pval`` column (from ``run_pirs(slope_pvals=True)``).

    Parameters
    ----------
    classifications : pd.DataFrame
    slope_pval_threshold : float
        BH-corrected p-value threshold drawn as a horizontal line (default 0.05).
    pirs_percentile : float
        PIRS percentile cutoff drawn as a vertical reference line (default 50).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    p_col = "slope_pval_bh" if _has(classifications, "slope_pval_bh") else "slope_pval"
    if not _has(classifications, p_col):
        raise ValueError(
            "'slope_pval' column is required. Call run_pirs(slope_pvals=True) first."
        )
    ax = _ax(ax)
    df = classifications.dropna(subset=["pirs_score", p_col])
    df = df.assign(neg_log_slope=_safe_neglog10(df[p_col]))
    pirs_cut = np.percentile(df["pirs_score"], pirs_percentile)

    _scatter_by_label(df, "pirs_score", "neg_log_slope", ax)

    ax.axhline(
        -np.log10(slope_pval_threshold),
        color="#333333",
        ls="--",
        lw=0.9,
        alpha=0.7,
        label=f"α = {slope_pval_threshold}",
    )
    ax.axvline(
        pirs_cut,
        color="#333333",
        ls=":",
        lw=0.9,
        alpha=0.7,
        label=f"PIRS p{int(pirs_percentile)}",
    )
    lbl_used = "slope_pval_bh" if p_col == "slope_pval_bh" else "slope_pval"
    _clip_axes_to_data(ax, df["pirs_score"], df["neg_log_slope"])
    ax.set_xlabel("PIRS score")
    ax.set_ylabel(f"−log₁₀({lbl_used})")
    ax.set_title(title)
    _label_legend(df["label"].unique() if "label" in df.columns else [], ax)
    sns.despine(ax=ax)
    return ax


def slope_vs_rhythm(
    classifications,
    slope_pval_threshold=0.05,
    emp_p_threshold=0.05,
    ax=None,
    title="Slope significance vs rhythmicity significance",
):
    """Scatter of −log₁₀(slope_pval_bh) vs −log₁₀(emp_p).

    Contrasts the two independent significance axes: linear drift (slope test)
    and circadian rhythmicity (BooteJTK empirical p-value).  The four quadrants
    correspond to the major label classes:

    * Top-right: significant slope **and** rhythmic — ``noisy_rhythmic`` or
      ``rhythmic`` depending on PIRS score.
    * Top-left: significant slope, not rhythmic — ``linear``.
    * Bottom-right: not sloped, rhythmic — ``rhythmic``.
    * Bottom-left: neither — ``constitutive`` or ``variable``.

    Requires ``slope_pval`` and ``emp_p`` columns.

    Parameters
    ----------
    classifications : pd.DataFrame
    slope_pval_threshold : float
        Slope significance threshold (default 0.05).
    emp_p_threshold : float
        Rhythm significance threshold (default 0.05).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    sp_col = "slope_pval_bh" if _has(classifications, "slope_pval_bh") else "slope_pval"
    if not _has(classifications, sp_col):
        raise ValueError(
            "'slope_pval' column is required. Call run_pirs(slope_pvals=True) first."
        )
    if not _has(classifications, "emp_p"):
        raise ValueError("'emp_p' column is required. Call run_bootjtk() first.")

    ax = _ax(ax)
    df = classifications.dropna(subset=[sp_col, "emp_p"])
    df = df.assign(
        neg_log_slope=_safe_neglog10(df[sp_col]),
        neg_log_emp_p=_safe_neglog10(df["emp_p"]),
    )

    _scatter_by_label(df, "neg_log_slope", "neg_log_emp_p", ax)

    ax.axvline(
        -np.log10(slope_pval_threshold),
        color="#333333",
        ls="--",
        lw=0.9,
        alpha=0.7,
        label=f"slope α = {slope_pval_threshold}",
    )
    ax.axhline(
        -np.log10(emp_p_threshold),
        color="#333333",
        ls=":",
        lw=0.9,
        alpha=0.7,
        label=f"rhythm α = {emp_p_threshold}",
    )
    lbl_used = "slope_pval_bh" if sp_col == "slope_pval_bh" else "slope_pval"
    _clip_axes_to_data(ax, df["neg_log_slope"], df["neg_log_emp_p"])
    ax.set_xlabel(f"−log₁₀({lbl_used})")
    ax.set_ylabel("−log₁₀(GammaBH)")
    ax.set_title(title)
    _label_legend(df["label"].unique() if "label" in df.columns else [], ax)
    sns.despine(ax=ax)
    return ax


def phase_wheel(
    classifications,
    labels=("rhythmic", "noisy_rhythmic"),
    ax=None,
    title="Phase distribution (rhythmic genes)",
):
    """Polar histogram of estimated phase angles for rhythmic genes.

    ``PhaseMean`` is in hours (0–24) and is converted to radians for the
    polar plot.  Only genes whose ``label`` is in *labels* are included.

    Requires ``phase_mean`` column (always present when ``run_bootjtk()`` has
    been called).

    Parameters
    ----------
    classifications : pd.DataFrame
    labels : tuple of str
        Labels to include (default: rhythmic and noisy_rhythmic).
    ax : matplotlib.axes.Axes, optional
        Must be a polar axes.  Created automatically if not provided.
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if not _has(classifications, "phase_mean"):
        raise ValueError("'phase_mean' column is required. Call run_bootjtk() first.")

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    df = classifications[classifications["label"].isin(labels)].dropna(
        subset=["phase_mean"]
    )
    if df.empty:
        df = classifications.dropna(subset=["phase_mean"])
        title = title + " (all genes — no rhythmic genes at current thresholds)"

    phases_rad = df["phase_mean"] * (2 * np.pi / 24)
    nbins = 12
    bins = np.linspace(0, 2 * np.pi, nbins + 1)
    counts, _ = np.histogram(phases_rad, bins=bins)
    widths = np.diff(bins)
    bars = ax.bar(
        bins[:-1],
        counts,
        width=widths,
        align="edge",
        alpha=0.7,
        color=LABEL_COLORS["rhythmic"],
        edgecolor="white",
    )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    hour_labels = [f"ZT{int(h):02d}" for h in np.linspace(0, 24, 8, endpoint=False)]
    ax.set_xticklabels(hour_labels, fontsize=9)

    # Show integer counts on the radial axis and label each bar
    max_count = max(counts) if counts.max() > 0 else 1
    ax.set_rmax(max_count * 1.30)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))
    ax.tick_params(axis="y", labelsize=8, labelcolor="#555555")
    for angle, count in zip(bins[:-1] + widths / 2, counts):
        if count > 0:
            ax.text(
                angle,
                count + max_count * 0.10,
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )

    ax.set_title(title, pad=15)
    return ax


def period_distribution(
    classifications,
    labels=("rhythmic", "noisy_rhythmic"),
    reference_period=24.0,
    ax=None,
    title="Period distribution (rhythmic genes)",
):
    """Histogram of estimated period lengths for rhythmic genes.

    A vertical dashed line marks *reference_period* (default 24 h).  Only
    genes whose ``label`` is in *labels* are included.

    Requires ``period_mean`` column (always present when ``run_bootjtk()`` has
    been called).

    Parameters
    ----------
    classifications : pd.DataFrame
    labels : tuple of str
    reference_period : float
        Reference period drawn as a dashed line (default 24.0).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if not _has(classifications, "period_mean"):
        raise ValueError("'period_mean' column is required. Call run_bootjtk() first.")

    ax = _ax(ax)

    # Collect data for the requested labels; fall back to all genes if none qualify
    label_subsets = {
        lbl: classifications[classifications["label"] == lbl]["period_mean"].dropna()
        for lbl in labels
    }
    has_data = any(not s.empty for s in label_subsets.values())
    if not has_data:
        all_genes = classifications["period_mean"].dropna()
        label_subsets = {"(all genes)": all_genes}
        title = title + " (all genes — no rhythmic genes at current thresholds)"

    all_periods = (
        pd.concat(list(label_subsets.values()))
        if label_subsets
        else pd.Series(dtype=float)
    )
    data_range = all_periods.max() - all_periods.min() if len(all_periods) > 1 else 0.0

    if data_range < 1.0:
        # Discrete or constant periods (e.g., all 24 h) — use a fixed window
        xlo, xhi = reference_period - 6, reference_period + 6
        bins = np.arange(xlo, xhi + 2, 2)  # 2-h bins across ±6 h window
    else:
        xlo = all_periods.min() - 1
        xhi = all_periods.max() + 1
        bins = min(20, max(5, int(data_range)))

    for lbl, sub in label_subsets.items():
        if not sub.empty:
            ax.hist(
                sub,
                bins=bins,
                color=LABEL_COLORS.get(lbl, "#8C8C8C"),
                alpha=0.6,
                label=lbl,
                edgecolor="white",
            )

    ax.axvline(
        reference_period,
        color="#333333",
        ls="--",
        lw=0.9,
        alpha=0.8,
        label=f"{reference_period:.0f} h",
    )
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("Period (h)")
    ax.set_ylabel("Gene count")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    sns.despine(ax=ax)
    return ax


def phase_amplitude_scatter(
    classifications,
    labels=("rhythmic", "noisy_rhythmic"),
    ax=None,
    title="Phase vs rhythm strength (rhythmic genes)",
):
    """Scatter of estimated phase angle vs rhythm strength for rhythmic genes.

    Phase (x-axis) is the estimated peak time in hours (0–24).  Rhythm
    strength (y-axis) is TauMean, the Kendall's tau correlation between the
    data and a cosine waveform — higher values indicate a stronger rhythmic
    signal.

    Requires ``phase_mean`` and ``tau_mean`` columns (present when
    ``run_bootjtk()`` has been called).

    Parameters
    ----------
    classifications : pd.DataFrame
    labels : tuple of str
        Labels to include (default: rhythmic and noisy_rhythmic).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if not _has(classifications, "phase_mean"):
        raise ValueError("'phase_mean' column is required. Call run_bootjtk() first.")

    ax = _ax(ax)
    df = classifications[classifications["label"].isin(labels)].dropna(
        subset=["phase_mean", "tau_mean"]
    )
    if df.empty:
        df = classifications.dropna(subset=["phase_mean", "tau_mean"])
        title = title + " (all genes — no rhythmic genes at current thresholds)"

    _scatter_by_label(df, "phase_mean", "tau_mean", ax, size=20, alpha=0.7)

    # Ensure the y-axis shows a minimum readable range even with few genes
    y_lo, y_hi = ax.get_ylim()
    if y_hi - y_lo < 0.15:
        mid = (y_lo + y_hi) / 2
        ax.set_ylim(max(0.0, mid - 0.075), min(1.15, mid + 0.075))

    ax.set_xlabel("Phase (h)")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xticklabels([f"ZT{h:02d}" for h in range(0, 25, 4)])
    ax.set_ylabel("Rhythm strength (TauMean)")
    ax.set_title(title)
    _label_legend(df["label"].unique(), ax)
    sns.despine(ax=ax)
    return ax


def top_constitutive_candidates(
    classifications,
    n_top=20,
    pirs_percentile=50,
    ax=None,
    title="Top constitutive gene candidates",
):
    """Ranked horizontal bar chart of the top-scoring constitutive gene candidates.

    Genes are ranked by PIRS score ascending (lower = more constitutive).
    When p-value columns are present, bar colours communicate evidence strength:

    * **Strong candidate** (``LABEL_COLORS['constitutive']``): constitutive
      label, significant PIRS p-value, and no significant linear slope.
    * **Moderate candidate** (lighter blue): significant PIRS p-value but a
      significant linear slope detected.
    * **Weak candidate** (pale blue): constitutive label but p-value not yet
      significant.
    * **Other labels** (their ``LABEL_COLORS``): genes with a low PIRS score
      that were classified differently (e.g. rhythmic).

    Parameters
    ----------
    classifications : pd.DataFrame
        Output of ``Classifier.classify()``.
    n_top : int
        Number of top candidates to display (default 20).
    pirs_percentile : float
        Percentile cutoff drawn as a vertical reference line (default 50).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ax(ax)
    df = classifications.dropna(subset=["pirs_score"])
    top = df.nsmallest(n_top, "pirs_score")

    all_scores = classifications["pirs_score"].dropna()
    pirs_cut = np.percentile(all_scores, pirs_percentile)

    p_col = "pval_bh" if _has(df, "pval_bh") else "pval" if _has(df, "pval") else None
    sp_col = (
        "slope_pval_bh"
        if _has(df, "slope_pval_bh")
        else "slope_pval"
        if _has(df, "slope_pval")
        else None
    )

    def _bar_color(row):
        lbl = row["label"] if "label" in row.index else "constitutive"
        if lbl != "constitutive":
            return LABEL_COLORS.get(lbl, "#8C8C8C")
        if p_col is None:
            return LABEL_COLORS["constitutive"]
        pval_sig = pd.notna(row[p_col]) and row[p_col] <= 0.05
        slope_ns = sp_col is None or (pd.notna(row[sp_col]) and row[sp_col] > 0.05)
        if pval_sig and slope_ns:
            return LABEL_COLORS["constitutive"]
        if pval_sig:
            return "#7BA7D4"
        return "#B0C4DE"

    colors = [_bar_color(row) for _, row in top.iterrows()]
    top_rev = top.iloc[::-1]
    colors_rev = colors[::-1]

    ax.barh(top_rev.index.tolist(), top_rev["pirs_score"].values, color=colors_rev)
    ax.axvline(
        pirs_cut,
        color="#333333",
        ls="--",
        lw=0.9,
        alpha=0.7,
        label=f"PIRS p{int(pirs_percentile)}",
    )
    ax.set_xlabel("PIRS score")
    ax.set_title(title)

    color_set = set(colors)
    legend_patches = []
    base_lbl = "constitutive" if p_col is None else "strong candidate"
    if LABEL_COLORS["constitutive"] in color_set:
        legend_patches.append(
            mpatches.Patch(color=LABEL_COLORS["constitutive"], label=base_lbl)
        )
    if "#7BA7D4" in color_set:
        legend_patches.append(mpatches.Patch(color="#7BA7D4", label="has linear slope"))
    if "#B0C4DE" in color_set:
        legend_patches.append(
            mpatches.Patch(color="#B0C4DE", label="not yet significant")
        )
    for lbl in _LABEL_ORDER:
        if lbl != "constitutive" and LABEL_COLORS.get(lbl) in color_set:
            legend_patches.append(mpatches.Patch(color=LABEL_COLORS[lbl], label=lbl))
    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower right", frameon=False, fontsize=8)

    sns.despine(ax=ax, left=True)
    return ax


def classification_summary(
    classifications,
    pirs_percentile=50,
    tau_threshold=0.5,
    emp_p_threshold=0.05,
    slope_pval_threshold=0.05,
    outpath=None,
):
    """Multi-panel summary figure for a classification result DataFrame.

    Panels are selected adaptively based on which optional columns are present:

    * **Always shown** (3 panels): label distribution, PIRS vs TauMean,
      PIRS score distribution.
    * **Requires ``emp_p``** (2 panels): volcano, TauMean vs GammaBH.
    * **Requires ``phase_mean``** (1 panel): phase wheel.
    * **Requires ``period_mean``** (1 panel): period distribution.
    * **Requires ``pval``** (1 panel): PIRS score vs temporal structure
      significance.
    * **Requires ``slope_pval`` and ``emp_p``** (1 panel): slope significance
      vs rhythm significance.

    Parameters
    ----------
    classifications : pd.DataFrame
        Output of ``Classifier.classify()``.
    pirs_percentile : float
        Passed to relevant individual plot functions (default 50).
    tau_threshold : float
        Passed to ``pirs_vs_tau`` and ``tau_pval_scatter`` (default 0.5).
    emp_p_threshold : float
        Passed to ``volcano`` and ``tau_pval_scatter`` (default 0.05).
    slope_pval_threshold : float
        Passed to ``slope_vs_rhythm`` and ``slope_pval_scatter`` (default 0.05).
    outpath : str or None
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    has_emp_p = _has(classifications, "emp_p")
    has_phase = _has(classifications, "phase_mean")
    has_period = _has(classifications, "period_mean")
    has_pval = _has(classifications, "pval") or _has(classifications, "pval_bh")
    has_slope = _has(classifications, "slope_pval") or _has(
        classifications, "slope_pval_bh"
    )
    has_slope_emp = has_slope and has_emp_p

    # Build ordered list of (title, callable) for each panel
    panels = [
        ("label_distribution", lambda ax: label_distribution(classifications, ax=ax)),
        (
            "pirs_vs_tau",
            lambda ax: pirs_vs_tau(
                classifications,
                pirs_percentile=pirs_percentile,
                tau_threshold=tau_threshold,
                ax=ax,
            ),
        ),
        (
            "pirs_score_distribution",
            lambda ax: pirs_score_distribution(
                classifications, pirs_percentile=pirs_percentile, ax=ax
            ),
        ),
        (
            "top_constitutive_candidates",
            lambda ax: top_constitutive_candidates(
                classifications, pirs_percentile=pirs_percentile, ax=ax
            ),
        ),
        (
            "threshold_sensitivity",
            lambda ax: threshold_sensitivity(
                classifications, pirs_percentile=pirs_percentile, ax=ax
            ),
        ),
    ]
    if has_emp_p:
        panels += [
            (
                "volcano",
                lambda ax: volcano(
                    classifications,
                    emp_p_threshold=emp_p_threshold,
                    pirs_percentile=pirs_percentile,
                    ax=ax,
                ),
            ),
            (
                "tau_pval_scatter",
                lambda ax: tau_pval_scatter(
                    classifications,
                    tau_threshold=tau_threshold,
                    emp_p_threshold=emp_p_threshold,
                    ax=ax,
                ),
            ),
        ]
    if has_pval:
        panels.append(
            ("pirs_pval_scatter", lambda ax: pirs_pval_scatter(classifications, ax=ax))
        )
    if has_slope:
        panels.append(
            (
                "slope_pval_scatter",
                lambda ax: slope_pval_scatter(
                    classifications,
                    slope_pval_threshold=slope_pval_threshold,
                    pirs_percentile=pirs_percentile,
                    ax=ax,
                ),
            )
        )
    if has_slope_emp:
        panels.append(
            (
                "slope_vs_rhythm",
                lambda ax: slope_vs_rhythm(
                    classifications,
                    slope_pval_threshold=slope_pval_threshold,
                    emp_p_threshold=emp_p_threshold,
                    ax=ax,
                ),
            )
        )
    if has_phase:
        panels.append(("phase_wheel", None))  # handled separately — polar axes
        panels.append(
            (
                "phase_amplitude_scatter",
                lambda ax: phase_amplitude_scatter(classifications, ax=ax),
            )
        )
    if has_period:
        panels.append(
            (
                "period_distribution",
                lambda ax: period_distribution(classifications, ax=ax),
            )
        )

    n = len(panels)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(5 * ncols, 4 * nrows))

    for i, (name, fn) in enumerate(panels, 1):
        if name == "phase_wheel":
            ax = fig.add_subplot(nrows, ncols, i, projection="polar")
            phase_wheel(classifications, ax=ax)
        else:
            ax = fig.add_subplot(nrows, ncols, i)
            fn(ax)

    fig.tight_layout(pad=1.5, h_pad=2.0, w_pad=1.5)
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
    return fig


def mean_expression_profiles(
    expression,
    classifications,
    labels=None,
    ax=None,
    title="Mean expression profile by label",
):
    """Mean ± SEM time-series expression profile for each expression label.

    Genes are grouped by their ``label`` from *classifications*, z-scored
    individually so that all genes contribute equally regardless of baseline
    level, and then averaged per label at each unique timepoint.  The shaded
    band shows ± 1 SEM across genes.

    Parameters
    ----------
    expression : pd.DataFrame
        Expression data with ZT/CT-prefixed sample columns (e.g. ``ZT02_1``),
        indexed by gene ID.
    classifications : pd.DataFrame
        Output of ``Classifier.classify()``, indexed by the same gene IDs.
    labels : list of str, optional
        Labels to include.  Defaults to all labels present in *classifications*.
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ax(ax)

    zt_cols, timepoints, unique_tp = _zt_timepoints(expression)

    # Average replicates → one value per gene per unique timepoint
    tp_means = pd.DataFrame(
        {
            int(tp): expression[
                [c for c, t in zip(zt_cols, timepoints) if t == tp]
            ].mean(axis=1)
            for tp in unique_tp
        },
        index=expression.index,
    )

    # Z-score each gene across its timepoint means so profiles are comparable
    row_mean = tp_means.mean(axis=1)
    row_std = tp_means.std(axis=1).replace(0, np.nan)
    tp_z = tp_means.sub(row_mean, axis=0).div(row_std, axis=0)

    common = tp_z.index.intersection(classifications.index)
    tp_z = tp_z.loc[common]
    clf = classifications.loc[common]

    if labels is None:
        labels = [l for l in _LABEL_ORDER if l in clf["label"].values]

    for lbl in labels:
        genes = clf[clf["label"] == lbl].index
        if len(genes) == 0:
            continue
        profiles = tp_z.loc[genes].astype(float)
        mean_p = profiles.mean(axis=0)
        sem_p = profiles.sem(axis=0)
        color = LABEL_COLORS.get(lbl, "#8C8C8C")
        ax.plot(unique_tp, mean_p.values, color=color, label=lbl, lw=1.5)
        ax.fill_between(
            unique_tp,
            (mean_p - sem_p).values,
            (mean_p + sem_p).values,
            color=color,
            alpha=0.15,
        )

    ax.axhline(0, color="#999999", ls=":", lw=0.8)
    ax.set_xlabel("Zeitgeber time (h)")
    ax.set_ylabel("Mean z-scored expression ± SEM")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    sns.despine(ax=ax)
    return ax


def gene_profile(
    expression,
    gene_id,
    classifications=None,
    color=None,
    ax=None,
    title=None,
):
    """Time-series profile for a single gene.

    Scatter-plots all replicate sample values with the per-timepoint mean as
    a line.  When *classifications* is provided the color is derived from the
    gene's label and the title is annotated with τ, phase, and PIRS score.

    Parameters
    ----------
    expression : pd.DataFrame
        Expression matrix with ZT/CT-prefixed sample columns.
    gene_id : str
        Row label in *expression*.
    classifications : pd.DataFrame, optional
        Output of ``Classifier.classify()``, indexed by the same gene IDs.
        Used to determine label color and to annotate the title with scores.
    color : str, optional
        Point/line color.  Derived from the label in *classifications* when
        omitted; falls back to ``LABEL_COLORS['constitutive']``.
    ax : matplotlib.axes.Axes, optional
    title : str, optional
        Defaults to *gene_id* with score annotations appended.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ax(ax)
    zt_cols, timepoints, unique_tp = _zt_timepoints(expression)

    if gene_id not in expression.index:
        raise ValueError(f"{gene_id!r} not found in expression index.")

    vals = expression.loc[gene_id, zt_cols].astype(float).values

    if color is None:
        if classifications is not None and gene_id in classifications.index:
            lbl = (
                classifications.at[gene_id, "label"]
                if "label" in classifications.columns
                else None
            )
            color = LABEL_COLORS.get(lbl, "#8C8C8C")
        else:
            color = LABEL_COLORS["constitutive"]

    ax.scatter(timepoints, vals, color=color, s=14, alpha=0.55, zorder=3)
    means = np.array(
        [
            vals[[i for i, t in enumerate(timepoints) if t == tp]].mean()
            for tp in unique_tp
        ]
    )
    ax.plot(unique_tp, means, color=color, lw=1.5, zorder=2)

    if title is None:
        title = gene_id
        if classifications is not None and gene_id in classifications.index:
            row = classifications.loc[gene_id]
            parts = []
            if "tau_mean" in row.index and pd.notna(row["tau_mean"]):
                parts.append(f"τ={row['tau_mean']:.2f}")
            if "phase_mean" in row.index and pd.notna(row["phase_mean"]):
                parts.append(f"φ={row['phase_mean']:.1f}h")
            if parts:
                title += "\n" + "  ".join(parts)
            if "pirs_score" in row.index and pd.notna(row["pirs_score"]):
                title += f"\nPIRS={row['pirs_score']:.3f}"

    ax.set_title(title)
    ax.set_xlabel("ZT (h)")
    ax.set_ylabel("Expression")
    ax.set_xticks(unique_tp)
    sns.despine(ax=ax)
    return ax


# Per-label column and sort direction for representative gene selection
_LABEL_SELECT_COL = {
    "constitutive": ("pirs_score", True),  # ascending PIRS → most stable
    "rhythmic": ("tau_mean", False),  # descending tau → clearest rhythm
    "noisy_rhythmic": ("tau_mean", False),
    "linear": ("slope_pval", True),  # ascending slope_pval → clearest trend
    "variable": ("pirs_score", False),  # descending PIRS → most variable
    "unclassified": (None, True),
}


def expression_heatmap(
    expression,
    classifications=None,
    labels=None,
    n_per_label=20,
    z_score=True,
    method="ward",
    cmap="RdBu_r",
    show_gene_labels=None,
    colorbar=True,
    ax=None,
    title="Expression heatmap",
):
    """Clustered heatmap of gene expression grouped by label.

    Genes are subsampled per label (most representative first), z-scored
    across timepoints, and sorted by within-group hierarchical clustering.
    A narrow color strip on the left side encodes the expression label.
    White lines separate label groups.

    Parameters
    ----------
    expression : pd.DataFrame
        Expression matrix with ZT/CT-prefixed sample columns, indexed by
        gene ID.
    classifications : pd.DataFrame, optional
        Output of ``Classifier.classify()``.  When provided, genes are
        grouped and color-coded by label.  If omitted, all genes are
        clustered together.
    labels : list of str, optional
        Labels to include, in display order.  Defaults to all labels in
        ``_LABEL_ORDER`` present in *classifications*.
    n_per_label : int
        Maximum genes per label.  Genes are ranked by the most
        informative score for each label (lowest PIRS for constitutive,
        highest TauMean for rhythmic, etc.).  Default 20.
    z_score : bool
        Z-score each gene across its timepoint means before plotting
        (default True).
    method : str
        Linkage method for within-group hierarchical clustering
        (default ``'ward'``).
    cmap : str
        Matplotlib colormap name (default ``'RdBu_r'``).
    show_gene_labels : bool or None
        Whether to draw gene-ID tick labels.  Auto-detects: shown when
        ≤ 40 genes, hidden otherwise.
    colorbar : bool
        Draw a colorbar for the z-score scale (default True).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
        The main heatmap axes.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.cluster.hierarchy import linkage, leaves_list

    ax = _ax(ax)
    zt_cols, timepoints, unique_tp = _zt_timepoints(expression)

    # Average replicates → one value per gene per unique timepoint
    tp_mat = pd.DataFrame(
        {
            int(tp): expression[
                [c for c, t in zip(zt_cols, timepoints) if t == tp]
            ].mean(axis=1)
            for tp in unique_tp
        },
        index=expression.index,
    ).astype(float)

    # -----------------------------------------------------------------------
    # Gene selection and ordering
    # -----------------------------------------------------------------------
    if classifications is not None:
        common = tp_mat.index.intersection(classifications.index)
        tp_mat = tp_mat.loc[common]
        clf = classifications.loc[common]

        if labels is None:
            labels = [l for l in _LABEL_ORDER if l in clf["label"].values]

        ordered_genes: list = []
        gene_label_list: list = []

        for lbl in labels:
            mask = clf["label"] == lbl
            genes = clf[mask].index.tolist()
            if not genes:
                continue

            n = min(n_per_label, len(genes))
            col, ascending = _LABEL_SELECT_COL.get(lbl, ("pirs_score", True))
            if col and col in clf.columns and clf[col].notna().any():
                ranked = clf.loc[genes, col].dropna()
                genes = (
                    ranked.nsmallest(n) if ascending else ranked.nlargest(n)
                ).index.tolist()
            else:
                genes = genes[:n]

            sub = tp_mat.loc[genes].values
            if len(sub) > 2:
                order = leaves_list(linkage(sub, method=method))
                genes = [genes[i] for i in order]

            ordered_genes.extend(genes)
            gene_label_list.extend([lbl] * len(genes))
    else:
        max_genes = n_per_label * len(_LABEL_ORDER)
        genes = tp_mat.index.tolist()
        if len(genes) > max_genes:
            step = max(1, len(genes) // max_genes)
            genes = genes[::step][:max_genes]
        sub = tp_mat.loc[genes].values
        if len(sub) > 2:
            order = leaves_list(linkage(sub, method=method))
            genes = [genes[i] for i in order]
        ordered_genes = genes
        gene_label_list = None

    if not ordered_genes:
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            "No genes to display",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#888888",
        )
        return ax

    mat = tp_mat.loc[ordered_genes].values
    n_genes, n_tp = mat.shape

    # -----------------------------------------------------------------------
    # Z-score each gene
    # -----------------------------------------------------------------------
    if z_score:
        row_mean = mat.mean(axis=1, keepdims=True)
        row_std = mat.std(axis=1, keepdims=True)
        row_std[row_std == 0] = 1.0
        mat = (mat - row_mean) / row_std

    vmax = float(np.nanpercentile(np.abs(mat), 99))
    vmax = max(vmax, 0.5)

    # -----------------------------------------------------------------------
    # Draw heatmap
    # -----------------------------------------------------------------------
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    # White lines between label groups
    if gene_label_list is not None:
        boundary = 0
        for lbl in labels or []:
            cnt = gene_label_list.count(lbl)
            boundary += cnt
            if 0 < boundary < n_genes:
                ax.axhline(boundary - 0.5, color="white", lw=1.2)

    # X-axis: unique timepoints
    ax.set_xticks(range(n_tp))
    ax.set_xticklabels([f"ZT{int(tp):02d}" for tp in unique_tp], fontsize=8)

    # Y-axis: gene IDs (auto-hide above threshold)
    show_labels = show_gene_labels if show_gene_labels is not None else (n_genes <= 40)
    if show_labels:
        ax.set_yticks(range(n_genes))
        ax.set_yticklabels(ordered_genes, fontsize=7)
    else:
        ax.set_yticks([])

    ax.set_title(title)

    # -----------------------------------------------------------------------
    # Label color strip and colorbar via AxesDivider
    # -----------------------------------------------------------------------
    divider = make_axes_locatable(ax)

    if gene_label_list is not None:
        cax_strip = divider.append_axes("left", size="4%", pad=0.05)
        rgb = np.array(
            [
                plt.matplotlib.colors.to_rgb(LABEL_COLORS.get(l, "#8C8C8C"))
                for l in gene_label_list
            ],
            dtype=float,
        ).reshape(n_genes, 1, 3)
        cax_strip.imshow(rgb, aspect="auto", interpolation="nearest")
        cax_strip.set_xticks([])
        cax_strip.set_yticks([])

    if colorbar:
        cax_cb = divider.append_axes("right", size="4%", pad=0.08)
        cb = plt.colorbar(im, cax=cax_cb)
        cb.set_label("z-score" if z_score else "expression", fontsize=8)

    return ax


def threshold_sensitivity(
    classifications,
    pirs_percentile=50,
    ax=None,
    title="PIRS score distribution by label (ECDF)",
):
    """Empirical CDFs of PIRS scores per expression label.

    Shows how each label's genes distribute across the PIRS score range so
    you can evaluate where the constitutive cutoff sits relative to each
    population.  A vertical dashed line marks the current *pirs_percentile*
    threshold.  Ideal separation: the constitutive curve rises steeply to
    the left of the cut while other labels rise slowly or not at all there.

    Parameters
    ----------
    classifications : pd.DataFrame
    pirs_percentile : float
        Current threshold to mark with a vertical line (default 50).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ax(ax)
    all_scores = classifications["pirs_score"].dropna()
    pirs_cut = np.percentile(all_scores, pirs_percentile)

    for lbl in _LABEL_ORDER:
        sub = classifications[classifications["label"] == lbl]["pirs_score"].dropna()
        if sub.empty:
            continue
        sorted_scores = np.sort(sub)
        ecdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        ax.step(
            sorted_scores,
            ecdf,
            color=LABEL_COLORS[lbl],
            label=lbl,
            lw=1.5,
            where="post",
        )

    ax.axvline(
        pirs_cut,
        color="#333333",
        ls="--",
        lw=0.9,
        alpha=0.8,
        label=f"p{int(pirs_percentile)} cut",
    )

    lo, hi = np.percentile(all_scores, [1, 99])
    margin = max((hi - lo) * 0.05, 0.05)
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("PIRS score")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    sns.despine(ax=ax)
    return ax
