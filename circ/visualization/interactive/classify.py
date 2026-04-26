"""Interactive (Plotly) visualizations for expression classification results."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from circ.visualization.classify import LABEL_COLORS, _LABEL_ORDER, _has, _safe_neglog10


def _scatter_traces(df, xcol, ycol):
    """One Scatter trace per label, coloured by LABEL_COLORS with gene-ID hover."""
    traces = []
    for lbl in _LABEL_ORDER:
        sub = df[df['label'] == lbl] if 'label' in df.columns else df
        if sub.empty:
            continue
        traces.append(go.Scatter(
            x=sub[xcol], y=sub[ycol],
            mode='markers',
            name=lbl,
            text=sub.index.tolist(),
            hovertemplate=(
                f'<b>%{{text}}</b><br>'
                f'{xcol}: %{{x:.4g}}<br>'
                f'{ycol}: %{{y:.4g}}<br>'
                f'label: {lbl}'
                '<extra></extra>'
            ),
            marker=dict(color=LABEL_COLORS.get(lbl, '#8C8C8C'), size=5, opacity=0.7),
        ))
    return traces


def label_distribution(classifications, title='Expression label counts'):
    """Interactive bar chart of gene counts per expression label.

    Parameters
    ----------
    classifications : pd.DataFrame
        Output of ``Classifier.classify()``.
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    present = [l for l in _LABEL_ORDER if l in classifications['label'].values]
    counts = (
        classifications['label']
        .value_counts()
        .reindex(present)
        .fillna(0)
        .astype(int)
    )
    fig = go.Figure(go.Bar(
        x=counts.values,
        y=counts.index.tolist(),
        orientation='h',
        marker_color=[LABEL_COLORS[l] for l in counts.index],
        text=counts.values,
        textposition='outside',
        hovertemplate='%{y}: %{x} genes<extra></extra>',
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Gene count',
        yaxis={'categoryorder': 'array', 'categoryarray': list(reversed(present))},
        showlegend=False,
    )
    return fig


def pirs_vs_tau(
    classifications,
    pirs_percentile=50,
    tau_threshold=0.5,
    title='PIRS score vs rhythmicity (TauMean)',
):
    """Interactive scatter of PIRS score vs BooteJTK TauMean.

    Hover over any point to see the gene ID, PIRS score, TauMean, and label.

    Parameters
    ----------
    classifications : pd.DataFrame
    pirs_percentile : float
    tau_threshold : float
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = classifications.dropna(subset=['pirs_score', 'tau_mean'])
    pirs_cut = np.percentile(df['pirs_score'], pirs_percentile)

    fig = go.Figure(_scatter_traces(df, 'pirs_score', 'tau_mean'))
    fig.add_vline(x=float(pirs_cut), line_dash='dash', line_color='#333333', opacity=0.7,
                  annotation_text=f'PIRS p{int(pirs_percentile)}',
                  annotation_position='top right')
    fig.add_hline(y=tau_threshold, line_dash='dot', line_color='#333333', opacity=0.7,
                  annotation_text=f'τ={tau_threshold}',
                  annotation_position='bottom right')
    fig.update_layout(title=title, xaxis_title='PIRS score', yaxis_title='TauMean',
                      legend_title='Label')
    return fig


def volcano(
    classifications,
    emp_p_threshold=0.05,
    pirs_percentile=50,
    title='PIRS score vs rhythmicity significance',
):
    """Interactive scatter of PIRS score vs −log₁₀(emp_p).

    Requires ``emp_p`` column (present when ``run_bootjtk()`` has been called).

    Parameters
    ----------
    classifications : pd.DataFrame
    emp_p_threshold : float
    pirs_percentile : float
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not _has(classifications, 'emp_p'):
        raise ValueError("'emp_p' column is required. Call run_bootjtk() first.")
    df = classifications.dropna(subset=['pirs_score', 'emp_p'])
    df = df.assign(neg_log_emp_p=_safe_neglog10(df['emp_p']))
    pirs_cut = np.percentile(df['pirs_score'], pirs_percentile)

    fig = go.Figure(_scatter_traces(df, 'pirs_score', 'neg_log_emp_p'))
    fig.add_vline(x=float(pirs_cut), line_dash='dot', line_color='#333333', opacity=0.7,
                  annotation_text=f'PIRS p{int(pirs_percentile)}',
                  annotation_position='top right')
    fig.add_hline(y=-np.log10(emp_p_threshold), line_dash='dash', line_color='#333333', opacity=0.7,
                  annotation_text=f'FDR={emp_p_threshold}',
                  annotation_position='bottom right')
    fig.update_layout(title=title, xaxis_title='PIRS score', yaxis_title='−log₁₀(GammaBH)',
                      legend_title='Label')
    return fig


def pirs_score_distribution(
    classifications,
    pirs_percentile=50,
    title='PIRS score distribution by label',
):
    """Interactive overlaid histograms of PIRS scores split by expression label.

    Parameters
    ----------
    classifications : pd.DataFrame
    pirs_percentile : float
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    all_scores = classifications['pirs_score'].dropna()
    pirs_cut = np.percentile(all_scores, pirs_percentile)

    fig = go.Figure()
    for lbl in _LABEL_ORDER:
        sub = classifications[classifications['label'] == lbl]['pirs_score'].dropna()
        if len(sub) < 2:
            continue
        fig.add_trace(go.Histogram(
            x=sub, name=lbl,
            marker_color=LABEL_COLORS.get(lbl, '#8C8C8C'),
            opacity=0.5,
            hovertemplate=f'label: {lbl}<br>PIRS score: %{{x:.4g}}<br>count: %{{y}}<extra></extra>',
        ))
    fig.add_vline(x=float(pirs_cut), line_dash='dash', line_color='#333333', opacity=0.8,
                  annotation_text=f'p{int(pirs_percentile)} cut',
                  annotation_position='top right')
    fig.update_layout(title=title, xaxis_title='PIRS score', yaxis_title='Count',
                      barmode='overlay', legend_title='Label')
    return fig


def tau_pval_scatter(
    classifications,
    tau_threshold=0.5,
    emp_p_threshold=0.05,
    title='Rhythmicity: TauMean vs GammaBH significance',
):
    """Interactive scatter of TauMean vs −log₁₀(emp_p).

    Requires ``emp_p`` column.

    Parameters
    ----------
    classifications : pd.DataFrame
    tau_threshold : float
    emp_p_threshold : float
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not _has(classifications, 'emp_p'):
        raise ValueError("'emp_p' column is required. Call run_bootjtk() first.")
    df = classifications.dropna(subset=['tau_mean', 'emp_p'])
    df = df.assign(neg_log_emp_p=_safe_neglog10(df['emp_p']))

    fig = go.Figure(_scatter_traces(df, 'tau_mean', 'neg_log_emp_p'))
    fig.add_vline(x=tau_threshold, line_dash='dash', line_color='#333333', opacity=0.7,
                  annotation_text=f'τ={tau_threshold}',
                  annotation_position='top right')
    fig.add_hline(y=-np.log10(emp_p_threshold), line_dash='dot', line_color='#333333', opacity=0.7,
                  annotation_text=f'FDR={emp_p_threshold}',
                  annotation_position='bottom right')
    fig.update_layout(title=title, xaxis_title='TauMean', yaxis_title='−log₁₀(GammaBH)',
                      legend_title='Label')
    return fig


def phase_wheel(
    classifications,
    labels=('rhythmic', 'noisy_rhythmic'),
    title='Phase distribution (rhythmic genes)',
):
    """Interactive polar bar chart of estimated phase angles for rhythmic genes.

    ``PhaseMean`` is in hours (0–24) and is converted to degrees for the polar
    plot.  Only genes whose ``label`` is in *labels* are included.

    Requires ``phase_mean`` column (always present when ``run_bootjtk()`` has
    been called).

    Parameters
    ----------
    classifications : pd.DataFrame
    labels : tuple of str
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not _has(classifications, 'phase_mean'):
        raise ValueError("'phase_mean' column is required. Call run_bootjtk() first.")

    df = classifications[classifications['label'].isin(labels)].dropna(subset=['phase_mean'])
    if df.empty:
        df = classifications.dropna(subset=['phase_mean'])
        title = title + ' (all genes — no rhythmic genes at current thresholds)'

    nbins = 12
    bin_size_h = 24 / nbins
    bin_edges = np.linspace(0, 24, nbins + 1)
    bin_centers_h = bin_edges[:-1] + bin_size_h / 2
    counts, _ = np.histogram(df['phase_mean'].values, bins=bin_edges)

    theta_deg = bin_centers_h * (360 / 24)
    width_deg = bin_size_h * (360 / 24)
    hover = [
        f'ZT{c - bin_size_h / 2:.0f}–ZT{c + bin_size_h / 2:.0f}: {n} genes'
        for c, n in zip(bin_centers_h, counts)
    ]

    fig = go.Figure(go.Barpolar(
        r=counts,
        theta=theta_deg,
        width=width_deg,
        marker_color='#6ACC65',
        marker_line_color='white',
        marker_line_width=1,
        opacity=0.7,
        hovertext=hover,
        hovertemplate='%{hovertext}<extra></extra>',
    ))
    fig.update_layout(
        title=title,
        polar=dict(
            angularaxis=dict(
                tickmode='array',
                tickvals=list(np.linspace(0, 360, 8, endpoint=False)),
                ticktext=[f'ZT{int(h):02d}' for h in np.linspace(0, 24, 8, endpoint=False)],
                direction='clockwise',
                rotation=90,
            ),
        ),
    )
    return fig


def period_distribution(
    classifications,
    labels=('rhythmic', 'noisy_rhythmic'),
    reference_period=24.0,
    title='Period distribution (rhythmic genes)',
):
    """Interactive histogram of estimated period lengths for rhythmic genes.

    A vertical dashed line marks *reference_period* (default 24 h).

    Requires ``period_mean`` column.

    Parameters
    ----------
    classifications : pd.DataFrame
    labels : tuple of str
    reference_period : float
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not _has(classifications, 'period_mean'):
        raise ValueError("'period_mean' column is required. Call run_bootjtk() first.")

    fig = go.Figure()
    any_data = False
    for lbl in labels:
        sub = classifications[classifications['label'] == lbl]['period_mean'].dropna()
        if sub.empty:
            continue
        any_data = True
        fig.add_trace(go.Histogram(
            x=sub, name=lbl,
            marker_color=LABEL_COLORS.get(lbl, '#8C8C8C'),
            opacity=0.6,
            hovertemplate=f'label: {lbl}<br>period: %{{x:.2f}} h<br>count: %{{y}}<extra></extra>',
        ))

    if not any_data:
        sub = classifications['period_mean'].dropna()
        fig.add_trace(go.Histogram(x=sub, name='all genes',
                                   marker_color='#8C8C8C', opacity=0.6))
        title = title + ' (all genes — no rhythmic genes at current thresholds)'

    fig.add_vline(x=reference_period, line_dash='dash', line_color='#333333', opacity=0.8,
                  annotation_text=f'{reference_period:.0f} h',
                  annotation_position='top right')
    fig.update_layout(title=title, xaxis_title='Period (h)', yaxis_title='Gene count',
                      barmode='overlay', legend_title='Label')
    return fig


def classification_summary(
    classifications,
    pirs_percentile=50,
    tau_threshold=0.5,
    emp_p_threshold=0.05,
    slope_pval_threshold=0.05,
):
    """Interactive multi-panel summary figure.

    Panels are selected adaptively based on which optional columns are present,
    matching the logic of ``circ.visualization.classification_summary``.

    Parameters
    ----------
    classifications : pd.DataFrame
        Output of ``Classifier.classify()``.
    pirs_percentile : float
    tau_threshold : float
    emp_p_threshold : float
    slope_pval_threshold : float

    Returns
    -------
    plotly.graph_objects.Figure
    """
    has_emp_p     = _has(classifications, 'emp_p')
    has_phase     = _has(classifications, 'phase_mean')
    has_period    = _has(classifications, 'period_mean')
    has_pval      = _has(classifications, 'pval') or _has(classifications, 'pval_bh')
    has_slope_emp = (
        (_has(classifications, 'slope_pval') or _has(classifications, 'slope_pval_bh'))
        and has_emp_p
    )

    panel_names = ['label_distribution', 'pirs_vs_tau', 'pirs_score_distribution']
    if has_emp_p:
        panel_names += ['volcano', 'tau_pval_scatter']
    if has_pval:
        panel_names.append('pirs_pval_scatter')
    if has_slope_emp:
        panel_names.append('slope_vs_rhythm')
    if has_phase:
        panel_names.append('phase_wheel')
    if has_period:
        panel_names.append('period_distribution')

    n = len(panel_names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    polar_cells = {
        (i // ncols + 1, i % ncols + 1)
        for i, name in enumerate(panel_names)
        if name == 'phase_wheel'
    }
    specs = [
        [
            {'type': 'polar'} if (r + 1, c + 1) in polar_cells else {'type': 'xy'}
            for c in range(ncols)
        ]
        for r in range(nrows)
    ]

    fig = make_subplots(
        rows=nrows, cols=ncols,
        specs=specs,
        subplot_titles=panel_names + [''] * (nrows * ncols - n),
    )

    # Resolve polar subplot names (e.g. 'polar', 'polar2') from the figure's
    # internal grid reference so Barpolar traces can be wired up without
    # using row=/col=, which Plotly 6 incorrectly maps to xaxis/yaxis.
    polar_refs = {}
    for i, name in enumerate(panel_names):
        if name == 'phase_wheel':
            r, c = i // ncols, i % ncols
            grid_cell = fig._grid_ref[r][c]
            if grid_cell:
                polar_refs[i] = grid_cell[0].layout_keys[0]  # e.g. 'polar' or 'polar2'

    kw = dict(
        pirs_percentile=pirs_percentile,
        tau_threshold=tau_threshold,
        emp_p_threshold=emp_p_threshold,
        slope_pval_threshold=slope_pval_threshold,
    )
    # Non-polar panels first — add_vline/add_hline in Plotly 6 iterates over
    # every trace in the figure and tries to set xaxis on each one, which
    # raises for go.Barpolar.  Adding phase_wheel last avoids this.
    for i, name in enumerate(panel_names):
        if name != 'phase_wheel':
            _add_panel(fig, name, classifications, i // ncols + 1, i % ncols + 1,
                       polar_ref=None, **kw)
    for i, name in enumerate(panel_names):
        if name == 'phase_wheel':
            _add_panel(fig, name, classifications, i // ncols + 1, i % ncols + 1,
                       polar_ref=polar_refs.get(i), **kw)

    fig.update_layout(
        height=350 * nrows,
        showlegend=False,
        title_text='Classification Summary',
    )
    return fig


def _add_panel(fig, name, df, row, col, polar_ref=None, **kw):
    """Add traces and threshold lines for one named panel to a subplot figure."""
    pp  = kw.get('pirs_percentile', 50)
    tau = kw.get('tau_threshold', 0.5)
    ep  = kw.get('emp_p_threshold', 0.05)
    sp  = kw.get('slope_pval_threshold', 0.05)

    def _traces(traces):
        for t in traces:
            t.showlegend = False
            fig.add_trace(t, row=row, col=col)

    if name == 'label_distribution':
        present = [l for l in _LABEL_ORDER if l in df['label'].values]
        counts = df['label'].value_counts().reindex(present).fillna(0).astype(int)
        fig.add_trace(go.Bar(
            x=counts.values, y=counts.index.tolist(), orientation='h',
            marker_color=[LABEL_COLORS[l] for l in counts.index],
            hovertemplate='%{y}: %{x} genes<extra></extra>',
            showlegend=False,
        ), row=row, col=col)

    elif name == 'pirs_vs_tau':
        sub = df.dropna(subset=['pirs_score', 'tau_mean'])
        pirs_cut = float(np.percentile(sub['pirs_score'], pp))
        _traces(_scatter_traces(sub, 'pirs_score', 'tau_mean'))
        fig.add_vline(x=pirs_cut, line_dash='dash', line_color='#333333', opacity=0.7,
                      row=row, col=col)
        fig.add_hline(y=tau, line_dash='dot', line_color='#333333', opacity=0.7,
                      row=row, col=col)

    elif name == 'pirs_score_distribution':
        all_scores = df['pirs_score'].dropna()
        pirs_cut = float(np.percentile(all_scores, pp))
        for lbl in _LABEL_ORDER:
            sub = df[df['label'] == lbl]['pirs_score'].dropna()
            if len(sub) < 2:
                continue
            fig.add_trace(go.Histogram(
                x=sub, name=lbl,
                marker_color=LABEL_COLORS.get(lbl, '#8C8C8C'),
                opacity=0.5, showlegend=False,
            ), row=row, col=col)
        fig.add_vline(x=pirs_cut, line_dash='dash', line_color='#333333', opacity=0.8,
                      row=row, col=col)

    elif name == 'volcano' and _has(df, 'emp_p'):
        sub = df.dropna(subset=['pirs_score', 'emp_p'])
        sub = sub.assign(neg_log_emp_p=_safe_neglog10(sub['emp_p']))
        pirs_cut = float(np.percentile(sub['pirs_score'], pp))
        _traces(_scatter_traces(sub, 'pirs_score', 'neg_log_emp_p'))
        fig.add_vline(x=pirs_cut, line_dash='dot', line_color='#333333', opacity=0.7,
                      row=row, col=col)
        fig.add_hline(y=-np.log10(ep), line_dash='dash', line_color='#333333', opacity=0.7,
                      row=row, col=col)

    elif name == 'tau_pval_scatter' and _has(df, 'emp_p'):
        sub = df.dropna(subset=['tau_mean', 'emp_p'])
        sub = sub.assign(neg_log_emp_p=_safe_neglog10(sub['emp_p']))
        _traces(_scatter_traces(sub, 'tau_mean', 'neg_log_emp_p'))
        fig.add_vline(x=tau, line_dash='dash', line_color='#333333', opacity=0.7,
                      row=row, col=col)
        fig.add_hline(y=-np.log10(ep), line_dash='dot', line_color='#333333', opacity=0.7,
                      row=row, col=col)

    elif name == 'pirs_pval_scatter':
        p_col = 'pval_bh' if _has(df, 'pval_bh') else 'pval'
        if _has(df, p_col):
            sub = df.dropna(subset=['pirs_score', p_col])
            sub = sub.assign(neg_log_p=_safe_neglog10(sub[p_col]))
            _traces(_scatter_traces(sub, 'pirs_score', 'neg_log_p'))
            fig.add_hline(y=-np.log10(0.05), line_dash='dash', line_color='#333333', opacity=0.7,
                          row=row, col=col)

    elif name == 'slope_vs_rhythm':
        sp_col = 'slope_pval_bh' if _has(df, 'slope_pval_bh') else 'slope_pval'
        if _has(df, sp_col) and _has(df, 'emp_p'):
            sub = df.dropna(subset=[sp_col, 'emp_p'])
            sub = sub.assign(
                neg_log_slope=_safe_neglog10(sub[sp_col]),
                neg_log_emp_p=_safe_neglog10(sub['emp_p']),
            )
            _traces(_scatter_traces(sub, 'neg_log_slope', 'neg_log_emp_p'))
            fig.add_vline(x=-np.log10(sp), line_dash='dash', line_color='#333333', opacity=0.7,
                          row=row, col=col)
            fig.add_hline(y=-np.log10(ep), line_dash='dot', line_color='#333333', opacity=0.7,
                          row=row, col=col)

    elif name == 'phase_wheel' and _has(df, 'phase_mean'):
        sub = df[df['label'].isin(['rhythmic', 'noisy_rhythmic'])].dropna(subset=['phase_mean'])
        if sub.empty:
            sub = df.dropna(subset=['phase_mean'])
        nbins = 12
        bin_size_h = 24 / nbins
        bin_edges = np.linspace(0, 24, nbins + 1)
        bin_centers_h = bin_edges[:-1] + bin_size_h / 2
        counts, _ = np.histogram(sub['phase_mean'].values, bins=bin_edges)
        # Set subplot explicitly — passing row=/col= to add_trace for a
        # Barpolar raises in Plotly 6 because it tries to assign xaxis.
        trace = go.Barpolar(
            r=counts,
            theta=bin_centers_h * (360 / 24),
            width=bin_size_h * (360 / 24),
            marker_color='#6ACC65',
            marker_line_color='white',
            marker_line_width=1,
            opacity=0.7,
            showlegend=False,
            subplot=polar_ref or 'polar',
        )
        fig.add_trace(trace)

    elif name == 'period_distribution' and _has(df, 'period_mean'):
        for lbl in ('rhythmic', 'noisy_rhythmic'):
            sub = df[df['label'] == lbl]['period_mean'].dropna()
            if sub.empty:
                continue
            fig.add_trace(go.Histogram(
                x=sub, name=lbl,
                marker_color=LABEL_COLORS.get(lbl, '#8C8C8C'),
                opacity=0.6, showlegend=False,
            ), row=row, col=col)
        fig.add_vline(x=24.0, line_dash='dash', line_color='#333333', opacity=0.8,
                      row=row, col=col)
