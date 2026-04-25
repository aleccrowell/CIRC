"""Visualizations for expression classification results from Classifier.classify()."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Consistent label colors used across every plot in this module
LABEL_COLORS = {
    'constitutive':  '#4878CF',
    'rhythmic':      '#6ACC65',
    'linear':        '#D65F5F',
    'variable':      '#B47CC7',
    'noisy_rhythmic': '#C4AD66',
    'unclassified':  '#8C8C8C',
}

_LABEL_ORDER = [
    'constitutive', 'rhythmic', 'linear',
    'variable', 'noisy_rhythmic', 'unclassified',
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ax(ax):
    return ax if ax is not None else plt.subplots()[1]


def _safe_neglog10(series, floor=1e-300):
    return -np.log10(np.maximum(series.clip(lower=floor), floor))


def _has(df, col):
    return col in df.columns and df[col].notna().any()


def _scatter_by_label(df, xcol, ycol, ax, size=18, alpha=0.65):
    for lbl in _LABEL_ORDER:
        sub = df[df['label'] == lbl] if 'label' in df.columns else df
        if not sub.empty:
            ax.scatter(
                sub[xcol], sub[ycol],
                color=LABEL_COLORS.get(lbl, '#8C8C8C'),
                s=size, alpha=alpha, label=lbl,
                rasterized=len(df) > 1000,
            )


def _label_legend(present, ax):
    patches = [
        mpatches.Patch(color=LABEL_COLORS[l], label=l)
        for l in _LABEL_ORDER if l in present
    ]
    if patches:
        ax.legend(handles=patches, loc='best', frameon=False, fontsize=8)


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

def label_distribution(classifications, ax=None, title='Expression label counts'):
    """Horizontal bar chart of gene counts per expression label.

    Parameters
    ----------
    classifications : pd.DataFrame
        Output of ``Classifier.classify()``.
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ax(ax)
    present = [l for l in _LABEL_ORDER if l in classifications['label'].values]
    counts = (
        classifications['label']
        .value_counts()
        .reindex(present)
        .fillna(0)
        .astype(int)
    )
    colors = [LABEL_COLORS[l] for l in counts.index]
    bars = ax.barh(counts.index, counts.values, color=colors)
    ax.bar_label(bars, fmt='%d', padding=3, fontsize=8)
    ax.set_xlabel('Gene count')
    ax.set_title(title)
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True)
    return ax


def pirs_vs_tau(
    classifications,
    pirs_percentile=50,
    tau_threshold=0.5,
    ax=None,
    title='PIRS score vs rhythmicity (TauMean)',
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
    df = classifications.dropna(subset=['pirs_score', 'tau_mean'])
    pirs_cut = np.percentile(df['pirs_score'], pirs_percentile)

    _scatter_by_label(df, 'pirs_score', 'tau_mean', ax)

    ax.axvline(pirs_cut, color='#333333', ls='--', lw=0.9, alpha=0.7,
               label=f'PIRS p{int(pirs_percentile)}')
    ax.axhline(tau_threshold, color='#333333', ls=':', lw=0.9, alpha=0.7,
               label=f'τ = {tau_threshold}')
    ax.set_xlabel('PIRS score')
    ax.set_ylabel('TauMean')
    ax.set_title(title)
    _label_legend(df['label'].unique() if 'label' in df.columns else [], ax)
    sns.despine(ax=ax)
    return ax


def volcano(
    classifications,
    emp_p_threshold=0.05,
    pirs_percentile=50,
    ax=None,
    title='PIRS score vs rhythmicity significance',
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
    if not _has(classifications, 'emp_p'):
        raise ValueError("'emp_p' column is required. Call run_bootjtk() first.")
    ax = _ax(ax)
    df = classifications.dropna(subset=['pirs_score', 'emp_p'])
    df = df.assign(neg_log_emp_p=_safe_neglog10(df['emp_p']))
    pirs_cut = np.percentile(df['pirs_score'], pirs_percentile)

    _scatter_by_label(df, 'pirs_score', 'neg_log_emp_p', ax)

    ax.axhline(-np.log10(emp_p_threshold), color='#333333', ls='--', lw=0.9, alpha=0.7,
               label=f'FDR = {emp_p_threshold}')
    ax.axvline(pirs_cut, color='#333333', ls=':', lw=0.9, alpha=0.7,
               label=f'PIRS p{int(pirs_percentile)}')
    _clip_axes_to_data(ax, df['pirs_score'], df['neg_log_emp_p'])
    ax.set_xlabel('PIRS score')
    ax.set_ylabel('−log₁₀(GammaBH)')
    ax.set_title(title)
    _label_legend(df['label'].unique() if 'label' in df.columns else [], ax)
    sns.despine(ax=ax)
    return ax


def pirs_score_distribution(
    classifications,
    pirs_percentile=50,
    ax=None,
    title='PIRS score distribution by label',
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
    all_scores = classifications['pirs_score'].dropna()
    pirs_cut = np.percentile(all_scores, pirs_percentile)
    for lbl in _LABEL_ORDER:
        sub = classifications[classifications['label'] == lbl]['pirs_score'].dropna()
        if len(sub) >= 3:
            sns.kdeplot(sub, ax=ax, color=LABEL_COLORS[lbl], label=lbl, fill=True,
                        alpha=0.3, linewidth=1.2)
    ax.axvline(pirs_cut, color='#333333', ls='--', lw=0.9, alpha=0.8,
               label=f'p{int(pirs_percentile)} cut')
    # Clip x-axis to 1st–99th percentile to avoid outlier compression
    lo, hi = np.percentile(all_scores, [1, 99])
    margin = max((hi - lo) * 0.1, 0.05)
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_xlabel('PIRS score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    sns.despine(ax=ax)
    return ax


def tau_pval_scatter(
    classifications,
    tau_threshold=0.5,
    emp_p_threshold=0.05,
    ax=None,
    title='Rhythmicity: TauMean vs GammaBH significance',
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
    if not _has(classifications, 'emp_p'):
        raise ValueError("'emp_p' column is required. Call run_bootjtk() first.")
    ax = _ax(ax)
    df = classifications.dropna(subset=['tau_mean', 'emp_p'])
    df = df.assign(neg_log_emp_p=_safe_neglog10(df['emp_p']))

    _scatter_by_label(df, 'tau_mean', 'neg_log_emp_p', ax)

    ax.axvline(tau_threshold, color='#333333', ls='--', lw=0.9, alpha=0.7,
               label=f'τ = {tau_threshold}')
    ax.axhline(-np.log10(emp_p_threshold), color='#333333', ls=':', lw=0.9, alpha=0.7,
               label=f'FDR = {emp_p_threshold}')
    _clip_axes_to_data(ax, df['tau_mean'], df['neg_log_emp_p'])
    ax.set_xlabel('TauMean')
    ax.set_ylabel('−log₁₀(GammaBH)')
    ax.set_title(title)
    _label_legend(df['label'].unique() if 'label' in df.columns else [], ax)
    sns.despine(ax=ax)
    return ax


def pirs_pval_scatter(
    classifications,
    pval_threshold=0.05,
    ax=None,
    title='PIRS score vs temporal structure significance',
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
    p_col = 'pval_bh' if _has(classifications, 'pval_bh') else 'pval'
    if not _has(classifications, p_col):
        raise ValueError(
            "'pval' column is required. Call run_pirs(pvals=True) first."
        )
    ax = _ax(ax)
    df = classifications.dropna(subset=['pirs_score', p_col])
    df = df.assign(neg_log_p=_safe_neglog10(df[p_col]))

    _scatter_by_label(df, 'pirs_score', 'neg_log_p', ax)

    ax.axhline(-np.log10(pval_threshold), color='#333333', ls='--', lw=0.9, alpha=0.7,
               label=f'α = {pval_threshold}')
    lbl_used = 'pval_bh' if p_col == 'pval_bh' else 'pval'
    _clip_axes_to_data(ax, df['pirs_score'], df['neg_log_p'])
    ax.set_xlabel('PIRS score')
    ax.set_ylabel(f'−log₁₀({lbl_used})')
    ax.set_title(title)
    _label_legend(df['label'].unique() if 'label' in df.columns else [], ax)
    sns.despine(ax=ax)
    return ax


def slope_pval_scatter(
    classifications,
    slope_pval_threshold=0.05,
    pirs_percentile=50,
    ax=None,
    title='PIRS score vs linear slope significance',
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
    p_col = 'slope_pval_bh' if _has(classifications, 'slope_pval_bh') else 'slope_pval'
    if not _has(classifications, p_col):
        raise ValueError(
            "'slope_pval' column is required. Call run_pirs(slope_pvals=True) first."
        )
    ax = _ax(ax)
    df = classifications.dropna(subset=['pirs_score', p_col])
    df = df.assign(neg_log_slope=_safe_neglog10(df[p_col]))
    pirs_cut = np.percentile(df['pirs_score'], pirs_percentile)

    _scatter_by_label(df, 'pirs_score', 'neg_log_slope', ax)

    ax.axhline(-np.log10(slope_pval_threshold), color='#333333', ls='--', lw=0.9, alpha=0.7,
               label=f'α = {slope_pval_threshold}')
    ax.axvline(pirs_cut, color='#333333', ls=':', lw=0.9, alpha=0.7,
               label=f'PIRS p{int(pirs_percentile)}')
    lbl_used = 'slope_pval_bh' if p_col == 'slope_pval_bh' else 'slope_pval'
    _clip_axes_to_data(ax, df['pirs_score'], df['neg_log_slope'])
    ax.set_xlabel('PIRS score')
    ax.set_ylabel(f'−log₁₀({lbl_used})')
    ax.set_title(title)
    _label_legend(df['label'].unique() if 'label' in df.columns else [], ax)
    sns.despine(ax=ax)
    return ax


def slope_vs_rhythm(
    classifications,
    slope_pval_threshold=0.05,
    emp_p_threshold=0.05,
    ax=None,
    title='Slope significance vs rhythmicity significance',
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
    sp_col = 'slope_pval_bh' if _has(classifications, 'slope_pval_bh') else 'slope_pval'
    if not _has(classifications, sp_col):
        raise ValueError(
            "'slope_pval' column is required. Call run_pirs(slope_pvals=True) first."
        )
    if not _has(classifications, 'emp_p'):
        raise ValueError("'emp_p' column is required. Call run_bootjtk() first.")

    ax = _ax(ax)
    df = classifications.dropna(subset=[sp_col, 'emp_p'])
    df = df.assign(
        neg_log_slope=_safe_neglog10(df[sp_col]),
        neg_log_emp_p=_safe_neglog10(df['emp_p']),
    )

    _scatter_by_label(df, 'neg_log_slope', 'neg_log_emp_p', ax)

    ax.axvline(-np.log10(slope_pval_threshold), color='#333333', ls='--', lw=0.9, alpha=0.7,
               label=f'slope α = {slope_pval_threshold}')
    ax.axhline(-np.log10(emp_p_threshold), color='#333333', ls=':', lw=0.9, alpha=0.7,
               label=f'rhythm α = {emp_p_threshold}')
    lbl_used = 'slope_pval_bh' if sp_col == 'slope_pval_bh' else 'slope_pval'
    _clip_axes_to_data(ax, df['neg_log_slope'], df['neg_log_emp_p'])
    ax.set_xlabel(f'−log₁₀({lbl_used})')
    ax.set_ylabel('−log₁₀(GammaBH)')
    ax.set_title(title)
    _label_legend(df['label'].unique() if 'label' in df.columns else [], ax)
    sns.despine(ax=ax)
    return ax


def phase_wheel(
    classifications,
    labels=('rhythmic', 'noisy_rhythmic'),
    ax=None,
    title='Phase distribution (rhythmic genes)',
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
    if not _has(classifications, 'phase_mean'):
        raise ValueError("'phase_mean' column is required. Call run_bootjtk() first.")

    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    df = classifications[classifications['label'].isin(labels)].dropna(subset=['phase_mean'])
    if df.empty:
        ax.set_title(title + ' (no data)')
        return ax

    phases_rad = df['phase_mean'] * (2 * np.pi / 24)
    nbins = 12
    bins = np.linspace(0, 2 * np.pi, nbins + 1)
    counts, _ = np.histogram(phases_rad, bins=bins)
    widths = np.diff(bins)
    bars = ax.bar(bins[:-1], counts, width=widths, align='edge', alpha=0.7,
                  color='#6ACC65', edgecolor='white')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    hour_labels = [f'ZT{int(h):02d}' for h in np.linspace(0, 24, 8, endpoint=False)]
    ax.set_xticklabels(hour_labels, fontsize=8)

    # Show integer counts on the radial axis and label each bar
    max_count = max(counts) if counts.max() > 0 else 1
    ax.set_rmax(max_count * 1.25)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))
    ax.tick_params(axis='y', labelsize=7, labelcolor='#555555')
    for angle, count in zip(bins[:-1] + widths / 2, counts):
        if count > 0:
            ax.text(angle, count + max_count * 0.07, str(count),
                    ha='center', va='bottom', fontsize=7, color='#333333')

    ax.set_title(title, pad=15)
    return ax


def period_distribution(
    classifications,
    labels=('rhythmic', 'noisy_rhythmic'),
    reference_period=24.0,
    ax=None,
    title='Period distribution (rhythmic genes)',
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
    if not _has(classifications, 'period_mean'):
        raise ValueError("'period_mean' column is required. Call run_bootjtk() first.")

    ax = _ax(ax)
    all_periods = pd.concat([
        classifications[classifications['label'] == lbl]['period_mean'].dropna()
        for lbl in labels
        if not classifications[classifications['label'] == lbl]['period_mean'].dropna().empty
    ]) if any(
        not classifications[classifications['label'] == lbl]['period_mean'].dropna().empty
        for lbl in labels
    ) else pd.Series(dtype=float)

    data_range = all_periods.max() - all_periods.min() if len(all_periods) > 1 else 0.0

    if data_range < 1.0:
        # Discrete or constant periods (e.g., all 24 h) — use a fixed window
        xlo, xhi = reference_period - 6, reference_period + 6
        bins = np.arange(xlo, xhi + 2, 2)  # 2-h bins across ±6 h window
    else:
        xlo = all_periods.min() - 1
        xhi = all_periods.max() + 1
        bins = min(20, max(5, int(data_range)))

    for lbl in labels:
        sub = classifications[classifications['label'] == lbl]['period_mean'].dropna()
        if not sub.empty:
            ax.hist(sub, bins=bins, color=LABEL_COLORS.get(lbl, '#8C8C8C'),
                    alpha=0.6, label=lbl, edgecolor='white')

    ax.axvline(reference_period, color='#333333', ls='--', lw=0.9, alpha=0.8,
               label=f'{reference_period:.0f} h')
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel('Period (h)')
    ax.set_ylabel('Gene count')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    sns.despine(ax=ax)
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
    has_emp_p     = _has(classifications, 'emp_p')
    has_phase     = _has(classifications, 'phase_mean')
    has_period    = _has(classifications, 'period_mean')
    has_pval      = _has(classifications, 'pval') or _has(classifications, 'pval_bh')
    has_slope_emp = (_has(classifications, 'slope_pval') or _has(classifications, 'slope_pval_bh')) and has_emp_p

    # Build ordered list of (title, callable) for each panel
    panels = [
        ('label_distribution', lambda ax: label_distribution(classifications, ax=ax)),
        ('pirs_vs_tau', lambda ax: pirs_vs_tau(
            classifications, pirs_percentile=pirs_percentile,
            tau_threshold=tau_threshold, ax=ax)),
        ('pirs_score_distribution', lambda ax: pirs_score_distribution(
            classifications, pirs_percentile=pirs_percentile, ax=ax)),
    ]
    if has_emp_p:
        panels += [
            ('volcano', lambda ax: volcano(
                classifications, emp_p_threshold=emp_p_threshold,
                pirs_percentile=pirs_percentile, ax=ax)),
            ('tau_pval_scatter', lambda ax: tau_pval_scatter(
                classifications, tau_threshold=tau_threshold,
                emp_p_threshold=emp_p_threshold, ax=ax)),
        ]
    if has_pval:
        panels.append(('pirs_pval_scatter', lambda ax: pirs_pval_scatter(
            classifications, ax=ax)))
    if has_slope_emp:
        panels.append(('slope_vs_rhythm', lambda ax: slope_vs_rhythm(
            classifications, slope_pval_threshold=slope_pval_threshold,
            emp_p_threshold=emp_p_threshold, ax=ax)))
    if has_phase:
        panels.append(('phase_wheel', None))  # handled separately — polar axes
    if has_period:
        panels.append(('period_distribution', lambda ax: period_distribution(
            classifications, ax=ax)))

    n = len(panels)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(5 * ncols, 4 * nrows))

    for i, (name, fn) in enumerate(panels, 1):
        if name == 'phase_wheel':
            ax = fig.add_subplot(nrows, ncols, i, projection='polar')
            phase_wheel(classifications, ax=ax)
        else:
            ax = fig.add_subplot(nrows, ncols, i)
            fn(ax)

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
    return fig
