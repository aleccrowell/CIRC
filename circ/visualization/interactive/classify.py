"""Interactive (Plotly) visualizations for expression classification results."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from circ.visualization.classify import LABEL_COLORS, _LABEL_ORDER, _has, _safe_neglog10


def _scatter_traces(df, xcol, ycol):
    """One Scatter trace per label, coloured by LABEL_COLORS with gene-ID hover."""
    traces = []
    for lbl in _LABEL_ORDER:
        sub = df[df["label"] == lbl] if "label" in df.columns else df
        if sub.empty:
            continue
        traces.append(
            go.Scatter(
                x=sub[xcol],
                y=sub[ycol],
                mode="markers",
                name=lbl,
                text=sub.index.tolist(),
                hovertemplate=(
                    f"<b>%{{text}}</b><br>"
                    f"{xcol}: %{{x:.4g}}<br>"
                    f"{ycol}: %{{y:.4g}}<br>"
                    f"label: {lbl}"
                    "<extra></extra>"
                ),
                marker=dict(
                    color=LABEL_COLORS.get(lbl, "#8C8C8C"), size=5, opacity=0.7
                ),
            )
        )
    return traces


def label_distribution(classifications, title="Expression label counts"):
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
    present = [l for l in _LABEL_ORDER if l in classifications["label"].values]
    counts = (
        classifications["label"].value_counts().reindex(present).fillna(0).astype(int)
    )
    fig = go.Figure(
        go.Bar(
            x=counts.values,
            y=counts.index.tolist(),
            orientation="h",
            marker_color=[LABEL_COLORS[l] for l in counts.index],
            text=counts.values,
            textposition="outside",
            hovertemplate="%{y}: %{x} genes<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Gene count",
        yaxis={"categoryorder": "array", "categoryarray": list(reversed(present))},
        showlegend=False,
    )
    return fig


def pirs_vs_tau(
    classifications,
    pirs_percentile=50,
    tau_threshold=0.5,
    title="PIRS score vs rhythmicity (TauMean)",
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
    df = classifications.dropna(subset=["pirs_score", "tau_mean"])
    pirs_cut = np.percentile(df["pirs_score"], pirs_percentile)

    fig = go.Figure(_scatter_traces(df, "pirs_score", "tau_mean"))
    fig.add_vline(
        x=float(pirs_cut),
        line_dash="dash",
        line_color="#333333",
        opacity=0.7,
        annotation_text=f"PIRS p{int(pirs_percentile)}",
        annotation_position="top right",
    )
    fig.add_hline(
        y=tau_threshold,
        line_dash="dot",
        line_color="#333333",
        opacity=0.7,
        annotation_text=f"τ={tau_threshold}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="PIRS score",
        yaxis_title="TauMean",
        legend_title="Label",
    )
    return fig


def volcano(
    classifications,
    emp_p_threshold=0.05,
    pirs_percentile=50,
    title="PIRS score vs rhythmicity significance",
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
    if not _has(classifications, "emp_p"):
        raise ValueError("'emp_p' column is required. Call run_bootjtk() first.")
    df = classifications.dropna(subset=["pirs_score", "emp_p"])
    df = df.assign(neg_log_emp_p=_safe_neglog10(df["emp_p"]))
    pirs_cut = np.percentile(df["pirs_score"], pirs_percentile)

    fig = go.Figure(_scatter_traces(df, "pirs_score", "neg_log_emp_p"))
    fig.add_vline(
        x=float(pirs_cut),
        line_dash="dot",
        line_color="#333333",
        opacity=0.7,
        annotation_text=f"PIRS p{int(pirs_percentile)}",
        annotation_position="top right",
    )
    fig.add_hline(
        y=-np.log10(emp_p_threshold),
        line_dash="dash",
        line_color="#333333",
        opacity=0.7,
        annotation_text=f"FDR={emp_p_threshold}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="PIRS score",
        yaxis_title="−log₁₀(GammaBH)",
        legend_title="Label",
    )
    return fig


def pirs_score_distribution(
    classifications,
    pirs_percentile=50,
    title="PIRS score distribution by label",
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
    all_scores = classifications["pirs_score"].dropna()
    pirs_cut = np.percentile(all_scores, pirs_percentile)

    fig = go.Figure()
    for lbl in _LABEL_ORDER:
        sub = classifications[classifications["label"] == lbl]["pirs_score"].dropna()
        if len(sub) < 2:
            continue
        fig.add_trace(
            go.Histogram(
                x=sub,
                name=lbl,
                marker_color=LABEL_COLORS.get(lbl, "#8C8C8C"),
                opacity=0.5,
                hovertemplate=f"label: {lbl}<br>PIRS score: %{{x:.4g}}<br>count: %{{y}}<extra></extra>",
            )
        )
    fig.add_vline(
        x=float(pirs_cut),
        line_dash="dash",
        line_color="#333333",
        opacity=0.8,
        annotation_text=f"p{int(pirs_percentile)} cut",
        annotation_position="top right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="PIRS score",
        yaxis_title="Count",
        barmode="overlay",
        legend_title="Label",
    )
    return fig


def tau_pval_scatter(
    classifications,
    tau_threshold=0.5,
    emp_p_threshold=0.05,
    title="Rhythmicity: TauMean vs GammaBH significance",
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
    if not _has(classifications, "emp_p"):
        raise ValueError("'emp_p' column is required. Call run_bootjtk() first.")
    df = classifications.dropna(subset=["tau_mean", "emp_p"])
    df = df.assign(neg_log_emp_p=_safe_neglog10(df["emp_p"]))

    fig = go.Figure(_scatter_traces(df, "tau_mean", "neg_log_emp_p"))
    fig.add_vline(
        x=tau_threshold,
        line_dash="dash",
        line_color="#333333",
        opacity=0.7,
        annotation_text=f"τ={tau_threshold}",
        annotation_position="top right",
    )
    fig.add_hline(
        y=-np.log10(emp_p_threshold),
        line_dash="dot",
        line_color="#333333",
        opacity=0.7,
        annotation_text=f"FDR={emp_p_threshold}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="TauMean",
        yaxis_title="−log₁₀(GammaBH)",
        legend_title="Label",
    )
    return fig


def phase_wheel(
    classifications,
    labels=("rhythmic", "noisy_rhythmic"),
    title="Phase distribution (rhythmic genes)",
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
    if not _has(classifications, "phase_mean"):
        raise ValueError("'phase_mean' column is required. Call run_bootjtk() first.")

    df = classifications[classifications["label"].isin(labels)].dropna(
        subset=["phase_mean"]
    )
    if df.empty:
        df = classifications.dropna(subset=["phase_mean"])
        title = title + " (all genes — no rhythmic genes at current thresholds)"

    nbins = 12
    bin_size_h = 24 / nbins
    bin_edges = np.linspace(0, 24, nbins + 1)
    bin_centers_h = bin_edges[:-1] + bin_size_h / 2
    counts, _ = np.histogram(df["phase_mean"].values, bins=bin_edges)

    theta_deg = bin_centers_h * (360 / 24)
    width_deg = bin_size_h * (360 / 24)
    hover = [
        f"ZT{c - bin_size_h / 2:.0f}–ZT{c + bin_size_h / 2:.0f}: {n} genes"
        for c, n in zip(bin_centers_h, counts)
    ]

    fig = go.Figure(
        go.Barpolar(
            r=counts,
            theta=theta_deg,
            width=width_deg,
            marker_color="#6ACC65",
            marker_line_color="white",
            marker_line_width=1,
            opacity=0.7,
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        polar=dict(
            angularaxis=dict(
                tickmode="array",
                tickvals=list(np.linspace(0, 360, 8, endpoint=False)),
                ticktext=[
                    f"ZT{int(h):02d}" for h in np.linspace(0, 24, 8, endpoint=False)
                ],
                direction="clockwise",
                rotation=90,
            ),
        ),
    )
    return fig


def period_distribution(
    classifications,
    labels=("rhythmic", "noisy_rhythmic"),
    reference_period=24.0,
    title="Period distribution (rhythmic genes)",
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
    if not _has(classifications, "period_mean"):
        raise ValueError("'period_mean' column is required. Call run_bootjtk() first.")

    fig = go.Figure()
    any_data = False
    for lbl in labels:
        sub = classifications[classifications["label"] == lbl]["period_mean"].dropna()
        if sub.empty:
            continue
        any_data = True
        fig.add_trace(
            go.Histogram(
                x=sub,
                name=lbl,
                marker_color=LABEL_COLORS.get(lbl, "#8C8C8C"),
                opacity=0.6,
                hovertemplate=f"label: {lbl}<br>period: %{{x:.2f}} h<br>count: %{{y}}<extra></extra>",
            )
        )

    if not any_data:
        sub = classifications["period_mean"].dropna()
        fig.add_trace(
            go.Histogram(x=sub, name="all genes", marker_color="#8C8C8C", opacity=0.6)
        )
        title = title + " (all genes — no rhythmic genes at current thresholds)"

    fig.add_vline(
        x=reference_period,
        line_dash="dash",
        line_color="#333333",
        opacity=0.8,
        annotation_text=f"{reference_period:.0f} h",
        annotation_position="top right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Period (h)",
        yaxis_title="Gene count",
        barmode="overlay",
        legend_title="Label",
    )
    return fig


def phase_amplitude_scatter(
    classifications,
    labels=("rhythmic", "noisy_rhythmic"),
    title="Phase vs rhythm strength (rhythmic genes)",
):
    """Interactive scatter of estimated phase angle vs rhythm strength.

    Hover over any point to see the gene ID, phase, TauMean, and label.

    Requires ``phase_mean`` and ``tau_mean`` columns.

    Parameters
    ----------
    classifications : pd.DataFrame
    labels : tuple of str
        Labels to include (default: rhythmic and noisy_rhythmic).
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not _has(classifications, "phase_mean"):
        raise ValueError("'phase_mean' column is required. Call run_bootjtk() first.")

    df = classifications[classifications["label"].isin(labels)].dropna(
        subset=["phase_mean", "tau_mean"]
    )
    if df.empty:
        df = classifications.dropna(subset=["phase_mean", "tau_mean"])
        title = title + " (all genes — no rhythmic genes at current thresholds)"

    traces = []
    for lbl in _LABEL_ORDER:
        sub = df[df["label"] == lbl] if "label" in df.columns else df
        if sub.empty:
            continue
        traces.append(
            go.Scatter(
                x=sub["phase_mean"],
                y=sub["tau_mean"],
                mode="markers",
                name=lbl,
                text=sub.index.tolist(),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "phase: %{x:.2f} h<br>"
                    "TauMean: %{y:.4g}<br>"
                    f"label: {lbl}"
                    "<extra></extra>"
                ),
                marker=dict(
                    color=LABEL_COLORS.get(lbl, "#8C8C8C"), size=5, opacity=0.7
                ),
            )
        )

    zt_vals = list(range(0, 25, 4))
    zt_texts = [f"ZT{h:02d}" for h in zt_vals]
    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Phase (h)",
            range=[0, 24],
            tickmode="array",
            tickvals=zt_vals,
            ticktext=zt_texts,
        ),
        yaxis_title="Rhythm strength (TauMean)",
        legend_title="Label",
    )
    return fig


def expression_heatmap(
    expression,
    classifications=None,
    labels=None,
    n_per_label=20,
    z_score=True,
    method="ward",
    title="Expression heatmap",
):
    """Interactive clustered heatmap of gene expression grouped by label.

    Hover over any cell to see the gene ID, timepoint, and z-score value.
    Genes are subsampled per label, z-scored, and sorted by within-group
    hierarchical clustering.  A label color annotation trace is drawn on
    the right y-axis.

    Parameters
    ----------
    expression : pd.DataFrame
        Expression matrix with ZT/CT-prefixed sample columns.
    classifications : pd.DataFrame, optional
        Output of ``Classifier.classify()``.
    labels : list of str, optional
    n_per_label : int
        Maximum genes per label (default 20).
    z_score : bool
        Z-score each gene before plotting (default True).
    method : str
        Linkage method for clustering (default ``'ward'``).
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    from circ.visualization.classify import (
        _zt_timepoints,
        _LABEL_ORDER,
        _LABEL_SELECT_COL,
        LABEL_COLORS,
    )
    from scipy.cluster.hierarchy import linkage, leaves_list

    zt_cols, timepoints, unique_tp = _zt_timepoints(expression)

    tp_mat = np.array(
        [
            expression[[c for c, t in zip(zt_cols, timepoints) if t == tp]]
            .mean(axis=1)
            .values
            for tp in unique_tp
        ]
    ).T  # shape (n_genes, n_tp)
    gene_index = expression.index.tolist()

    if classifications is not None:
        common = [g for g in gene_index if g in classifications.index]
        clf = classifications.loc[common]
        mat_sub = tp_mat[[gene_index.index(g) for g in common], :]
        gene_index = common

        if labels is None:
            labels = [l for l in _LABEL_ORDER if l in clf["label"].values]

        ordered_genes: list = []
        gene_label_list: list = []

        for lbl in labels:
            genes = clf[clf["label"] == lbl].index.tolist()
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

            idx_in_sub = [common.index(g) for g in genes]
            sub = mat_sub[idx_in_sub, :]
            if len(sub) > 2:
                order = leaves_list(linkage(sub, method=method))
                genes = [genes[i] for i in order]

            ordered_genes.extend(genes)
            gene_label_list.extend([lbl] * len(genes))

        final_idx = [common.index(g) for g in ordered_genes]
        mat = mat_sub[final_idx, :]
    else:
        max_genes = n_per_label * len(_LABEL_ORDER)
        if len(gene_index) > max_genes:
            step = max(1, len(gene_index) // max_genes)
            gene_index = gene_index[::step][:max_genes]
            tp_mat = tp_mat[[expression.index.tolist().index(g) for g in gene_index], :]
        if len(gene_index) > 2:
            order = leaves_list(linkage(tp_mat.astype(float), method=method))
            gene_index = [gene_index[i] for i in order]
            tp_mat = tp_mat[order, :]
        ordered_genes = gene_index
        gene_label_list = None
        mat = tp_mat

    mat = mat.astype(float)

    if z_score:
        row_mean = mat.mean(axis=1, keepdims=True)
        row_std = mat.std(axis=1, keepdims=True)
        row_std[row_std == 0] = 1.0
        mat = (mat - row_mean) / row_std

    vmax = float(np.percentile(np.abs(mat), 99))
    vmax = max(vmax, 0.5)

    x_labels = [f"ZT{int(tp):02d}" for tp in unique_tp]

    # Build hover text matrix
    hover = np.array(
        [
            [
                f"<b>{gene}</b><br>ZT: {x_labels[j]}<br>z-score: {mat[i, j]:.2f}"
                for j in range(mat.shape[1])
            ]
            for i, gene in enumerate(ordered_genes)
        ]
    )

    fig = go.Figure(
        go.Heatmap(
            z=mat,
            x=x_labels,
            y=ordered_genes,
            colorscale="RdBu",
            reversescale=True,
            zmin=-vmax,
            zmax=vmax,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            colorbar=dict(title="z-score" if z_score else "expression"),
        )
    )

    # Label color annotation traces on the right y-axis (as a thin bar)
    if gene_label_list is not None:
        fig.update_layout(
            yaxis2=dict(overlaying="y", side="right", showticklabels=False)
        )
        for lbl in labels or []:
            idxs = [i for i, l in enumerate(gene_label_list) if l == lbl]
            if not idxs:
                continue
            fig.add_trace(
                go.Bar(
                    x=[0.5] * len(idxs),
                    y=[ordered_genes[i] for i in idxs],
                    orientation="h",
                    name=lbl,
                    marker_color=LABEL_COLORS.get(lbl, "#8C8C8C"),
                    width=0.9,
                    yaxis="y2",
                    showlegend=True,
                    hovertemplate=f"label: {lbl}<extra></extra>",
                    visible=True,
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Timepoint",
        yaxis_title="Gene",
        yaxis=dict(autorange="reversed"),
        height=max(400, len(ordered_genes) * 14 + 100),
    )
    return fig


def top_constitutive_candidates(
    classifications,
    n_top=20,
    pirs_percentile=50,
    title="Top constitutive gene candidates",
):
    """Interactive ranked horizontal bar chart of the top constitutive gene candidates.

    Genes are ranked by PIRS score ascending (lower = more constitutive).
    Hover over any bar to see PIRS score, p-values, and the assigned label.
    Bar colours communicate evidence strength — see the static version for
    full tier documentation.

    Parameters
    ----------
    classifications : pd.DataFrame
        Output of ``Classifier.classify()``.
    n_top : int
        Number of top candidates to display (default 20).
    pirs_percentile : float
        Percentile cutoff drawn as a vertical reference line (default 50).
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    from circ.visualization.classify import LABEL_COLORS as _LC
    import pandas as _pd

    df = classifications.dropna(subset=["pirs_score"])
    top = df.nsmallest(n_top, "pirs_score")

    all_scores = classifications["pirs_score"].dropna()
    pirs_cut = float(np.percentile(all_scores, pirs_percentile))

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
            return _LC.get(lbl, "#8C8C8C")
        if p_col is None:
            return _LC["constitutive"]
        pval_sig = _pd.notna(row[p_col]) and row[p_col] <= 0.05
        slope_ns = sp_col is None or (_pd.notna(row[sp_col]) and row[sp_col] > 0.05)
        if pval_sig and slope_ns:
            return _LC["constitutive"]
        if pval_sig:
            return "#7BA7D4"
        return "#B0C4DE"

    hover_texts = []
    for gene_id, row in top.iterrows():
        txt = f"<b>{gene_id}</b><br>PIRS score: {row['pirs_score']:.4g}"
        if p_col and _pd.notna(row.get(p_col, float("nan"))):
            txt += f"<br>{p_col}: {row[p_col]:.3g}"
        if sp_col and _pd.notna(row.get(sp_col, float("nan"))):
            txt += f"<br>{sp_col}: {row[sp_col]:.3g}"
        if "label" in row.index:
            txt += f"<br>label: {row['label']}"
        hover_texts.append(txt)

    colors = [_bar_color(row) for _, row in top.iterrows()]
    top_rev = top.iloc[::-1]
    colors_rev = colors[::-1]
    hover_rev = hover_texts[::-1]

    fig = go.Figure(
        go.Bar(
            x=top_rev["pirs_score"].values,
            y=top_rev.index.tolist(),
            orientation="h",
            marker_color=colors_rev,
            hovertext=hover_rev,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )
    fig.add_vline(
        x=pirs_cut,
        line_dash="dash",
        line_color="#333333",
        opacity=0.7,
        annotation_text=f"PIRS p{int(pirs_percentile)}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="PIRS score",
        showlegend=False,
    )
    return fig


def pirs_pval_scatter(
    classifications,
    pval_threshold=0.05,
    title="PIRS score vs temporal structure significance",
):
    """Interactive scatter of PIRS score vs −log₁₀(pval_bh).

    Requires ``pval`` column (from ``run_pirs(pvals=True)``).

    Parameters
    ----------
    classifications : pd.DataFrame
    pval_threshold : float
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    p_col = "pval_bh" if _has(classifications, "pval_bh") else "pval"
    if not _has(classifications, p_col):
        raise ValueError("'pval' column is required. Call run_pirs(pvals=True) first.")
    df = classifications.dropna(subset=["pirs_score", p_col])
    df = df.assign(neg_log_p=_safe_neglog10(df[p_col]))

    fig = go.Figure(_scatter_traces(df, "pirs_score", "neg_log_p"))
    fig.add_hline(
        y=-np.log10(pval_threshold),
        line_dash="dash",
        line_color="#333333",
        opacity=0.7,
        annotation_text=f"α={pval_threshold}",
        annotation_position="bottom right",
    )
    lbl = "pval_bh" if p_col == "pval_bh" else "pval"
    fig.update_layout(
        title=title,
        xaxis_title="PIRS score",
        yaxis_title=f"−log₁₀({lbl})",
        legend_title="Label",
    )
    return fig


def slope_pval_scatter(
    classifications,
    slope_pval_threshold=0.05,
    pirs_percentile=50,
    title="PIRS score vs linear slope significance",
):
    """Interactive scatter of PIRS score vs −log₁₀(slope_pval_bh).

    Requires ``slope_pval`` column (from ``run_pirs(slope_pvals=True)``).

    Parameters
    ----------
    classifications : pd.DataFrame
    slope_pval_threshold : float
    pirs_percentile : float
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    sp_col = "slope_pval_bh" if _has(classifications, "slope_pval_bh") else "slope_pval"
    if not _has(classifications, sp_col):
        raise ValueError(
            "'slope_pval' column is required. Call run_pirs(slope_pvals=True) first."
        )
    df = classifications.dropna(subset=["pirs_score", sp_col])
    df = df.assign(neg_log_slope=_safe_neglog10(df[sp_col]))
    pirs_cut = float(np.percentile(df["pirs_score"], pirs_percentile))

    fig = go.Figure(_scatter_traces(df, "pirs_score", "neg_log_slope"))
    fig.add_hline(
        y=-np.log10(slope_pval_threshold),
        line_dash="dash",
        line_color="#333333",
        opacity=0.7,
        annotation_text=f"α={slope_pval_threshold}",
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=pirs_cut,
        line_dash="dot",
        line_color="#333333",
        opacity=0.7,
        annotation_text=f"PIRS p{int(pirs_percentile)}",
        annotation_position="top right",
    )
    lbl = "slope_pval_bh" if sp_col == "slope_pval_bh" else "slope_pval"
    fig.update_layout(
        title=title,
        xaxis_title="PIRS score",
        yaxis_title=f"−log₁₀({lbl})",
        legend_title="Label",
    )
    return fig


def slope_vs_rhythm(
    classifications,
    slope_pval_threshold=0.05,
    emp_p_threshold=0.05,
    title="Slope significance vs rhythmicity significance",
):
    """Interactive scatter of −log₁₀(slope_pval_bh) vs −log₁₀(emp_p).

    Requires ``slope_pval`` and ``emp_p`` columns.

    Parameters
    ----------
    classifications : pd.DataFrame
    slope_pval_threshold : float
    emp_p_threshold : float
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    sp_col = "slope_pval_bh" if _has(classifications, "slope_pval_bh") else "slope_pval"
    if not _has(classifications, sp_col):
        raise ValueError(
            "'slope_pval' column is required. Call run_pirs(slope_pvals=True) first."
        )
    if not _has(classifications, "emp_p"):
        raise ValueError("'emp_p' column is required. Call run_bootjtk() first.")
    df = classifications.dropna(subset=[sp_col, "emp_p"])
    df = df.assign(
        neg_log_slope=_safe_neglog10(df[sp_col]),
        neg_log_emp_p=_safe_neglog10(df["emp_p"]),
    )
    lbl = "slope_pval_bh" if sp_col == "slope_pval_bh" else "slope_pval"

    fig = go.Figure(_scatter_traces(df, "neg_log_slope", "neg_log_emp_p"))
    fig.add_vline(
        x=-np.log10(slope_pval_threshold),
        line_dash="dash",
        line_color="#333333",
        opacity=0.7,
        annotation_text=f"slope α={slope_pval_threshold}",
        annotation_position="top right",
    )
    fig.add_hline(
        y=-np.log10(emp_p_threshold),
        line_dash="dot",
        line_color="#333333",
        opacity=0.7,
        annotation_text=f"rhythm α={emp_p_threshold}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=title,
        xaxis_title=f"−log₁₀({lbl})",
        yaxis_title="−log₁₀(GammaBH)",
        legend_title="Label",
    )
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
    has_emp_p = _has(classifications, "emp_p")
    has_phase = _has(classifications, "phase_mean")
    has_period = _has(classifications, "period_mean")
    has_pval = _has(classifications, "pval") or _has(classifications, "pval_bh")
    has_slope_emp = (
        _has(classifications, "slope_pval") or _has(classifications, "slope_pval_bh")
    ) and has_emp_p

    panel_names = [
        "label_distribution",
        "pirs_vs_tau",
        "pirs_score_distribution",
        "top_constitutive_candidates",
    ]
    if has_emp_p:
        panel_names += ["volcano", "tau_pval_scatter"]
    if has_pval:
        panel_names.append("pirs_pval_scatter")
    if has_slope_emp:
        panel_names.append("slope_vs_rhythm")
    if has_phase:
        panel_names.append("phase_wheel")
        panel_names.append("phase_amplitude_scatter")
    if has_period:
        panel_names.append("period_distribution")

    n = len(panel_names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    polar_cells = {
        (i // ncols + 1, i % ncols + 1)
        for i, name in enumerate(panel_names)
        if name == "phase_wheel"
    }
    specs = [
        [
            {"type": "polar"} if (r + 1, c + 1) in polar_cells else {"type": "xy"}
            for c in range(ncols)
        ]
        for r in range(nrows)
    ]

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=specs,
        subplot_titles=panel_names + [""] * (nrows * ncols - n),
    )

    # Resolve polar subplot names (e.g. 'polar', 'polar2') from the figure's
    # internal grid reference so Barpolar traces can be wired up without
    # using row=/col=, which Plotly 6 incorrectly maps to xaxis/yaxis.
    polar_refs = {}
    for i, name in enumerate(panel_names):
        if name == "phase_wheel":
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
        if name != "phase_wheel":
            _add_panel(
                fig,
                name,
                classifications,
                i // ncols + 1,
                i % ncols + 1,
                polar_ref=None,
                **kw,
            )
    for i, name in enumerate(panel_names):
        if name == "phase_wheel":
            _add_panel(
                fig,
                name,
                classifications,
                i // ncols + 1,
                i % ncols + 1,
                polar_ref=polar_refs.get(i),
                **kw,
            )

    fig.update_layout(
        height=350 * nrows,
        showlegend=False,
        title_text="Classification Summary",
    )
    return fig


def _add_panel(fig, name, df, row, col, polar_ref=None, **kw):
    """Add traces and threshold lines for one named panel to a subplot figure."""
    pp = kw.get("pirs_percentile", 50)
    tau = kw.get("tau_threshold", 0.5)
    ep = kw.get("emp_p_threshold", 0.05)
    sp = kw.get("slope_pval_threshold", 0.05)

    def _traces(traces):
        for t in traces:
            t.showlegend = False
            fig.add_trace(t, row=row, col=col)

    if name == "label_distribution":
        present = [l for l in _LABEL_ORDER if l in df["label"].values]
        counts = df["label"].value_counts().reindex(present).fillna(0).astype(int)
        fig.add_trace(
            go.Bar(
                x=counts.values,
                y=counts.index.tolist(),
                orientation="h",
                marker_color=[LABEL_COLORS[l] for l in counts.index],
                hovertemplate="%{y}: %{x} genes<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    elif name == "pirs_vs_tau":
        sub = df.dropna(subset=["pirs_score", "tau_mean"])
        pirs_cut = float(np.percentile(sub["pirs_score"], pp))
        _traces(_scatter_traces(sub, "pirs_score", "tau_mean"))
        fig.add_vline(
            x=pirs_cut,
            line_dash="dash",
            line_color="#333333",
            opacity=0.7,
            row=row,
            col=col,
        )
        fig.add_hline(
            y=tau, line_dash="dot", line_color="#333333", opacity=0.7, row=row, col=col
        )

    elif name == "pirs_score_distribution":
        all_scores = df["pirs_score"].dropna()
        pirs_cut = float(np.percentile(all_scores, pp))
        for lbl in _LABEL_ORDER:
            sub = df[df["label"] == lbl]["pirs_score"].dropna()
            if len(sub) < 2:
                continue
            fig.add_trace(
                go.Histogram(
                    x=sub,
                    name=lbl,
                    marker_color=LABEL_COLORS.get(lbl, "#8C8C8C"),
                    opacity=0.5,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        fig.add_vline(
            x=pirs_cut,
            line_dash="dash",
            line_color="#333333",
            opacity=0.8,
            row=row,
            col=col,
        )

    elif name == "volcano" and _has(df, "emp_p"):
        sub = df.dropna(subset=["pirs_score", "emp_p"])
        sub = sub.assign(neg_log_emp_p=_safe_neglog10(sub["emp_p"]))
        pirs_cut = float(np.percentile(sub["pirs_score"], pp))
        _traces(_scatter_traces(sub, "pirs_score", "neg_log_emp_p"))
        fig.add_vline(
            x=pirs_cut,
            line_dash="dot",
            line_color="#333333",
            opacity=0.7,
            row=row,
            col=col,
        )
        fig.add_hline(
            y=-np.log10(ep),
            line_dash="dash",
            line_color="#333333",
            opacity=0.7,
            row=row,
            col=col,
        )

    elif name == "tau_pval_scatter" and _has(df, "emp_p"):
        sub = df.dropna(subset=["tau_mean", "emp_p"])
        sub = sub.assign(neg_log_emp_p=_safe_neglog10(sub["emp_p"]))
        _traces(_scatter_traces(sub, "tau_mean", "neg_log_emp_p"))
        fig.add_vline(
            x=tau, line_dash="dash", line_color="#333333", opacity=0.7, row=row, col=col
        )
        fig.add_hline(
            y=-np.log10(ep),
            line_dash="dot",
            line_color="#333333",
            opacity=0.7,
            row=row,
            col=col,
        )

    elif name == "pirs_pval_scatter":
        p_col = "pval_bh" if _has(df, "pval_bh") else "pval"
        if _has(df, p_col):
            sub = df.dropna(subset=["pirs_score", p_col])
            sub = sub.assign(neg_log_p=_safe_neglog10(sub[p_col]))
            _traces(_scatter_traces(sub, "pirs_score", "neg_log_p"))
            fig.add_hline(
                y=-np.log10(0.05),
                line_dash="dash",
                line_color="#333333",
                opacity=0.7,
                row=row,
                col=col,
            )

    elif name == "slope_vs_rhythm":
        sp_col = "slope_pval_bh" if _has(df, "slope_pval_bh") else "slope_pval"
        if _has(df, sp_col) and _has(df, "emp_p"):
            sub = df.dropna(subset=[sp_col, "emp_p"])
            sub = sub.assign(
                neg_log_slope=_safe_neglog10(sub[sp_col]),
                neg_log_emp_p=_safe_neglog10(sub["emp_p"]),
            )
            _traces(_scatter_traces(sub, "neg_log_slope", "neg_log_emp_p"))
            fig.add_vline(
                x=-np.log10(sp),
                line_dash="dash",
                line_color="#333333",
                opacity=0.7,
                row=row,
                col=col,
            )
            fig.add_hline(
                y=-np.log10(ep),
                line_dash="dot",
                line_color="#333333",
                opacity=0.7,
                row=row,
                col=col,
            )

    elif name == "phase_wheel" and _has(df, "phase_mean"):
        sub = df[df["label"].isin(["rhythmic", "noisy_rhythmic"])].dropna(
            subset=["phase_mean"]
        )
        if sub.empty:
            sub = df.dropna(subset=["phase_mean"])
        nbins = 12
        bin_size_h = 24 / nbins
        bin_edges = np.linspace(0, 24, nbins + 1)
        bin_centers_h = bin_edges[:-1] + bin_size_h / 2
        counts, _ = np.histogram(sub["phase_mean"].values, bins=bin_edges)
        # Set subplot explicitly — passing row=/col= to add_trace for a
        # Barpolar raises in Plotly 6 because it tries to assign xaxis.
        trace = go.Barpolar(
            r=counts,
            theta=bin_centers_h * (360 / 24),
            width=bin_size_h * (360 / 24),
            marker_color="#6ACC65",
            marker_line_color="white",
            marker_line_width=1,
            opacity=0.7,
            showlegend=False,
            subplot=polar_ref or "polar",
        )
        fig.add_trace(trace)

    elif name == "slope_pval_scatter":
        sp_col = "slope_pval_bh" if _has(df, "slope_pval_bh") else "slope_pval"
        if _has(df, sp_col):
            sub = df.dropna(subset=["pirs_score", sp_col])
            pirs_cut = float(np.percentile(sub["pirs_score"], pp))
            sub = sub.assign(neg_log_slope=_safe_neglog10(sub[sp_col]))
            _traces(_scatter_traces(sub, "pirs_score", "neg_log_slope"))
            fig.add_hline(
                y=-np.log10(sp),
                line_dash="dash",
                line_color="#333333",
                opacity=0.7,
                row=row,
                col=col,
            )
            fig.add_vline(
                x=pirs_cut,
                line_dash="dot",
                line_color="#333333",
                opacity=0.7,
                row=row,
                col=col,
            )

    elif name == "phase_amplitude_scatter" and _has(df, "phase_mean"):
        sub = df[df["label"].isin(["rhythmic", "noisy_rhythmic"])].dropna(
            subset=["phase_mean", "tau_mean"]
        )
        if sub.empty:
            sub = df.dropna(subset=["phase_mean", "tau_mean"])
        _traces(_scatter_traces(sub, "phase_mean", "tau_mean"))

    elif name == "top_constitutive_candidates":
        import pandas as _pd

        top = df.dropna(subset=["pirs_score"]).nsmallest(20, "pirs_score")
        all_sc = df["pirs_score"].dropna()
        pc = float(np.percentile(all_sc, pp))
        p_col = (
            "pval_bh" if _has(df, "pval_bh") else "pval" if _has(df, "pval") else None
        )
        sp_col2 = (
            "slope_pval_bh"
            if _has(df, "slope_pval_bh")
            else "slope_pval"
            if _has(df, "slope_pval")
            else None
        )

        def _bc(r):
            lbl = r["label"] if "label" in r.index else "constitutive"
            if lbl != "constitutive":
                return LABEL_COLORS.get(lbl, "#8C8C8C")
            if p_col is None:
                return LABEL_COLORS["constitutive"]
            ps = _pd.notna(r[p_col]) and r[p_col] <= 0.05
            sn = sp_col2 is None or (_pd.notna(r[sp_col2]) and r[sp_col2] > 0.05)
            if ps and sn:
                return LABEL_COLORS["constitutive"]
            return "#7BA7D4" if ps else "#B0C4DE"

        colors2 = [_bc(r) for _, r in top.iterrows()]
        top_rev = top.iloc[::-1]
        fig.add_trace(
            go.Bar(
                x=top_rev["pirs_score"].values,
                y=top_rev.index.tolist(),
                orientation="h",
                marker_color=colors2[::-1],
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_vline(
            x=pc, line_dash="dash", line_color="#333333", opacity=0.7, row=row, col=col
        )

    elif name == "period_distribution" and _has(df, "period_mean"):
        for lbl in ("rhythmic", "noisy_rhythmic"):
            sub = df[df["label"] == lbl]["period_mean"].dropna()
            if sub.empty:
                continue
            fig.add_trace(
                go.Histogram(
                    x=sub,
                    name=lbl,
                    marker_color=LABEL_COLORS.get(lbl, "#8C8C8C"),
                    opacity=0.6,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        fig.add_vline(
            x=24.0,
            line_dash="dash",
            line_color="#333333",
            opacity=0.8,
            row=row,
            col=col,
        )
