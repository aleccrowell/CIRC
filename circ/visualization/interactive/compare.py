"""Interactive (Plotly) visualizations for condition comparison results."""

import numpy as np
import plotly.graph_objects as go

from circ.visualization.compare import _STATUS_COLORS, _STATUS_ORDER


def rhythmicity_shift_scatter(
    comparison,
    alpha=0.05,
    title="Rhythmicity shift: TauMean per condition",
):
    """Interactive scatter of tau_mean_A vs tau_mean_B, colored by status.

    Each point is a shared gene.  Hover shows gene ID, tau values, Δ tau,
    rhythmicity status, and FDR p-value when available.  When ``tau_padj``
    is present, significant genes (padj < *alpha*) are drawn larger and
    more opaque than the non-significant background.

    Parameters
    ----------
    comparison : pd.DataFrame
        Output of :func:`~circ.compare.compare_conditions`.
    alpha : float
        FDR threshold for distinguishing significant genes (default 0.05).
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    has_padj = "tau_padj" in comparison.columns
    lim = max(comparison["tau_mean_A"].max(), comparison["tau_mean_B"].max()) * 1.05

    extra_cols = [c for c in ["delta_tau", "tau_padj"] if c in comparison.columns]

    def _hover(sub):
        parts = [
            "<b>%{text}</b>",
            "TauMean A: %{x:.3f}",
            "TauMean B: %{y:.3f}",
        ]
        for i, col in enumerate(extra_cols):
            if col == "delta_tau":
                parts.append("Δ tau: %{customdata[" + str(i) + "]:+.3f}")
            elif col == "tau_padj":
                parts.append("FDR: %{customdata[" + str(i) + "]:.3g}")
        return "<br>".join(parts) + "<extra></extra>"

    traces = []
    for status in _STATUS_ORDER:
        sub = comparison[comparison["rhythmicity_status"] == status]
        if sub.empty:
            continue
        color = _STATUS_COLORS[status]
        hovertemplate = _hover(sub)

        if has_padj:
            sig = sub["tau_padj"] < alpha
            for mask, name_suffix, size, opacity, show_legend in [
                (sig, f" (FDR<{alpha})", 7, 0.85, True),
                (~sig, "", 4, 0.35, not sig.any()),
            ]:
                if not mask.any():
                    continue
                s = sub[mask]
                traces.append(
                    go.Scatter(
                        x=s["tau_mean_A"],
                        y=s["tau_mean_B"],
                        mode="markers",
                        name=f"{status}{name_suffix}",
                        text=s.index.tolist(),
                        customdata=s[extra_cols].values if extra_cols else None,
                        hovertemplate=hovertemplate,
                        marker=dict(
                            color=color,
                            size=size,
                            opacity=opacity,
                            line=dict(width=0.5, color="white") if name_suffix else dict(width=0),
                        ),
                        legendgroup=status,
                        showlegend=show_legend,
                    )
                )
        else:
            traces.append(
                go.Scatter(
                    x=sub["tau_mean_A"],
                    y=sub["tau_mean_B"],
                    mode="markers",
                    name=f"{status} (n={len(sub)})",
                    text=sub.index.tolist(),
                    customdata=sub[extra_cols].values if extra_cols else None,
                    hovertemplate=hovertemplate,
                    marker=dict(color=color, size=5, opacity=0.7),
                )
            )

    traces.append(
        go.Scatter(
            x=[0, lim],
            y=[0, lim],
            mode="lines",
            line=dict(color="#999999", dash="dash", width=1),
            name="y = x",
            hoverinfo="skip",
        )
    )

    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        xaxis_title="TauMean — condition A",
        yaxis_title="TauMean — condition B",
        xaxis=dict(range=[0, lim]),
        yaxis=dict(range=[0, lim]),
    )
    return fig


def delta_tau_volcano(
    comparison,
    alpha=0.05,
    title="Rhythmicity change: Δ TauMean vs significance",
):
    """Interactive volcano plot of Δ TauMean vs −log₁₀(tau_padj).

    Hover shows gene ID, Δ tau, p-value, condition tau values, and status.

    Requires ``tau_padj`` column.

    Parameters
    ----------
    comparison : pd.DataFrame
        Output of :func:`~circ.compare.compare_conditions`.
    alpha : float
        FDR threshold drawn as a horizontal reference line (default 0.05).
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if "tau_padj" not in comparison.columns:
        raise ValueError(
            "'tau_padj' column missing — run compare_conditions() with result "
            "DataFrames that include tau_std and n_boots."
        )

    df = comparison.dropna(subset=["delta_tau", "tau_padj"]).copy()
    df["_neg_log_p"] = -np.log10(df["tau_padj"].clip(lower=1e-300))

    extra_cols = [c for c in ["tau_mean_A", "tau_mean_B", "tau_padj"] if c in df.columns]

    hover_parts = [
        "<b>%{text}</b>",
        "Δ tau: %{x:+.3f}",
        "−log₁₀(FDR): %{y:.2f}",
    ]
    for i, col in enumerate(extra_cols):
        label = {"tau_mean_A": "TauMean A", "tau_mean_B": "TauMean B", "tau_padj": "FDR"}[col]
        fmt = ".3f" if col.startswith("tau_mean") else ".3g"
        hover_parts.append(f"{label}: %{{customdata[{i}]:{fmt}}}")
    hovertemplate = "<br>".join(hover_parts) + "<extra></extra>"

    traces = []
    for status in _STATUS_ORDER:
        sub = df[df["rhythmicity_status"] == status]
        if sub.empty:
            continue
        traces.append(
            go.Scatter(
                x=sub["delta_tau"],
                y=sub["_neg_log_p"],
                mode="markers",
                name=status,
                text=sub.index.tolist(),
                customdata=sub[extra_cols].values if extra_cols else None,
                hovertemplate=hovertemplate,
                marker=dict(color=_STATUS_COLORS[status], size=5, opacity=0.7),
            )
        )

    fig = go.Figure(traces)
    fig.add_hline(
        y=-np.log10(alpha),
        line_dash="dash",
        line_color="#333333",
        annotation_text=f"FDR = {alpha}",
        annotation_position="top right",
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#999999", opacity=0.5)
    fig.update_layout(
        title=title,
        xaxis_title="Δ TauMean (B − A)",
        yaxis_title="−log₁₀(tau_padj)",
    )
    return fig
