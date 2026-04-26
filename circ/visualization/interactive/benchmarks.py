"""Interactive (Plotly) benchmark visualizations: PR curves and ROC curves."""
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc,
)


_COLORS = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA',
    '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
]

_TASK_NAMES = {
    ('pirs_score', 'Const'):      'PIRS score → constitutive',
    ('tau_mean',   'Circadian'):  'TauMean → circadian',
    ('emp_p',      'Circadian'):  'GammaBH p-val → circadian',
    ('pval',       'Const'):      'PIRS p-val → constitutive',
    ('pval_bh',    'Const'):      'PIRS p-val (BH) → constitutive',
    ('slope_pval', 'Linear'):     'slope p-val → linear',
    ('slope_pval_bh', 'Linear'):  'slope p-val (BH) → linear',
}


def classification_pr(
    classifications,
    true_classes,
    ground_truth_col='Const',
    score_col='pirs_score',
    invert_score=True,
    title=None,
):
    """Interactive PR curve comparing ``Classifier`` output against simulation ground truth.

    Hover shows recall, precision, and the threshold value at each point.

    Parameters
    ----------
    classifications : pd.DataFrame
        Output of ``Classifier.classify()``, indexed by gene ID.
    true_classes : pd.DataFrame
        Simulation ground-truth table with binary columns (e.g. ``Const``).
    ground_truth_col : str
    score_col : str
    invert_score : bool
        If ``True`` (default), invert the score so smaller values rank higher
        (appropriate for PIRS where low = constitutive).
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    merged = classifications[[score_col]].join(
        true_classes[[ground_truth_col]], how='inner'
    ).dropna()

    fig = go.Figure()

    if merged.empty:
        fig.update_layout(title=(title or 'PR curve') + ' (no matched genes)')
        return fig

    scores = -merged[score_col].values if invert_score else merged[score_col].values
    precision, recall, thresholds = precision_recall_curve(
        merged[ground_truth_col].values, scores, pos_label=1
    )
    baseline = merged[ground_truth_col].mean()
    ap = average_precision_score(merged[ground_truth_col].values, scores)

    # precision_recall_curve returns len(thresholds) + 1 precision/recall points
    thresh_padded = np.append(thresholds, np.nan)
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'{score_col} (AP={ap:.3f})',
        line=dict(color='#4878CF', width=2),
        customdata=thresh_padded.reshape(-1, 1),
        hovertemplate=(
            'recall: %{x:.3f}<br>'
            'precision: %{y:.3f}<br>'
            'threshold: %{customdata[0]:.4g}'
            '<extra></extra>'
        ),
    ))
    fig.add_hline(y=baseline, line_dash='dot', line_color='#CC4444', opacity=0.8,
                  annotation_text='random', annotation_position='bottom right')
    fig.update_layout(
        title=title or f'PR curve: {ground_truth_col}',
        xaxis=dict(title='Recall', range=[0, 1]),
        yaxis=dict(title='Precision', range=[0, 1.05]),
    )
    return fig


def classification_roc(
    classifications,
    true_classes,
    tasks=None,
    title='ROC: Classifier scores vs ground truth',
):
    """Interactive ROC curves for multiple binary classification tasks.

    Hover shows FPR, TPR, and the threshold value at each point.

    Parameters
    ----------
    classifications : pd.DataFrame
        Output of ``Classifier.classify()``.
    true_classes : pd.DataFrame
        Binary ground-truth table with columns like ``Const``, ``Circadian``,
        ``Linear``.
    tasks : list of (str, str, bool) or None
        Each entry is ``(score_col, truth_col, invert)``.  Defaults to sensible
        combinations based on available columns.
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if tasks is None:
        tasks = []
        if 'pirs_score' in classifications.columns and 'Const' in true_classes.columns:
            tasks.append(('pirs_score', 'Const', True))
        if 'tau_mean' in classifications.columns and 'Circadian' in true_classes.columns:
            tasks.append(('tau_mean', 'Circadian', False))
        if 'emp_p' in classifications.columns and 'Circadian' in true_classes.columns:
            tasks.append(('emp_p', 'Circadian', True))
        if 'pval' in classifications.columns and 'Const' in true_classes.columns:
            tasks.append(('pval', 'Const', True))
        if 'slope_pval' in classifications.columns and 'Linear' in true_classes.columns:
            tasks.append(('slope_pval', 'Linear', True))

    fig = go.Figure()
    for i, (score_col, truth_col, invert) in enumerate(tasks):
        merged = classifications[[score_col]].join(
            true_classes[[truth_col]], how='inner'
        ).dropna()
        if merged.empty:
            continue
        scores = -merged[score_col].values if invert else merged[score_col].values
        fpr, tpr, thresholds = roc_curve(merged[truth_col].values, scores, pos_label=1)
        auc_val = auc(fpr, tpr)
        label = _TASK_NAMES.get((score_col, truth_col), f'{score_col} → {truth_col}')

        thresh_padded = np.append(thresholds, np.nan)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{label}  (AUC={auc_val:.3f})',
            line=dict(color=_COLORS[i % len(_COLORS)], width=2),
            customdata=thresh_padded.reshape(-1, 1),
            hovertemplate=(
                'FPR: %{x:.3f}<br>'
                'TPR: %{y:.3f}<br>'
                'threshold: %{customdata[0]:.4g}'
                '<extra></extra>'
            ),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='random (AUC=0.5)',
        line=dict(color='#999999', dash='dash', width=1),
        hoverinfo='skip',
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title='False positive rate', range=[0, 1]),
        yaxis=dict(title='True positive rate', range=[0, 1.05]),
        legend=dict(x=0.01, y=0.01, xanchor='left', yanchor='bottom'),
    )
    return fig
