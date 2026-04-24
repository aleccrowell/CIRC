"""Benchmark visualizations: PR curves, ROC curves, and AUC comparison.

The standalone functions here are used by the ``analyze`` classes in
``circ.pirs.simulations`` and ``circ.limbr.simulations``, and can also be
called directly with pre-computed data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ax(ax):
    return ax if ax is not None else plt.subplots()[1]


# ---------------------------------------------------------------------------
# Precision-recall
# ---------------------------------------------------------------------------

def pr_curve(curves, baseline=None, ax=None, outpath=None):
    """Plot one or more precision-recall curves from a pre-computed DataFrame.

    Parameters
    ----------
    curves : pd.DataFrame
        Columns: ``precision``, ``recall``, ``method``, ``rep`` (rep is used
        as the ``units`` dimension for ``sns.lineplot``).
    baseline : float or None
        If given, draw a horizontal dashed line at this precision level (the
        random-classifier baseline equals the positive-class fraction).
    ax : matplotlib.axes.Axes, optional
    outpath : str or None
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ax(ax)
    palette = sns.color_palette('muted', n_colors=curves['method'].nunique())
    sns.lineplot(
        data=curves,
        x='recall', y='precision',
        hue='method', units='rep',
        palette=palette,
        estimator=None,
        linewidth=0.8,
        ax=ax,
    )
    if baseline is not None:
        ax.axhline(baseline, color='#CC4444', ls=':', lw=0.9,
                   label='random classifier')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision–Recall comparison')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0),
              frameon=False, fontsize=8)
    sns.despine(ax=ax)
    if outpath:
        ax.get_figure().savefig(outpath, dpi=150, bbox_inches='tight')
    return ax


def roc_curve_plot(merged_dict, ax=None, outpath=None):
    """Plot one or more ROC curves.

    Parameters
    ----------
    merged_dict : dict[str, pd.DataFrame]
        ``{tag: df}`` where each ``df`` has a binary truth column and a score
        column.  The truth column must be named ``truth`` or ``Circadian``, and
        the score column ``score`` or ``GammaBH``.
    ax : matplotlib.axes.Axes, optional
    outpath : str or None

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ax(ax)
    for tag, df in merged_dict.items():
        truth_col = 'truth' if 'truth' in df.columns else 'Circadian'
        score_col = 'score' if 'score' in df.columns else 'GammaBH'
        fpr, tpr, _ = roc_curve(df[truth_col].values, 1 - df[score_col].values, pos_label=1)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{tag}  (AUC={auc_val:.3f})')
    ax.plot([0, 1], [0, 1], color='#999999', ls='--', lw=0.8, label='random')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC comparison')
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    sns.despine(ax=ax)
    if outpath:
        ax.get_figure().savefig(outpath, dpi=150, bbox_inches='tight')
    return ax


def roc_auc(merged_dict):
    """Compute area under the ROC curve for each method.

    Parameters
    ----------
    merged_dict : dict[str, pd.DataFrame]
        Same format as :func:`roc_curve_plot`.

    Returns
    -------
    dict[str, float]
        ``{tag: auc_value}``.
    """
    out = {}
    for tag, df in merged_dict.items():
        truth_col = 'truth' if 'truth' in df.columns else 'Circadian'
        score_col = 'score' if 'score' in df.columns else 'GammaBH'
        fpr, tpr, _ = roc_curve(df[truth_col].values, 1 - df[score_col].values, pos_label=1)
        out[tag] = auc(fpr, tpr)
    return out


# ---------------------------------------------------------------------------
# Classification benchmarking (Classifier output + simulation ground truth)
# ---------------------------------------------------------------------------

def classification_pr(
    classifications,
    true_classes,
    ground_truth_col='Const',
    score_col='pirs_score',
    invert_score=True,
    ax=None,
    title=None,
):
    """PR curve comparing ``Classifier`` output against simulation ground truth.

    Parameters
    ----------
    classifications : pd.DataFrame
        Output of ``Classifier.classify()``, indexed by gene ID.
    true_classes : pd.DataFrame
        Simulation ground-truth table (from ``simulate.write_output``), with
        binary columns such as ``Const``, ``Circadian``, and ``Linear``.
    ground_truth_col : str
        Which binary column in *true_classes* to treat as positive (default
        ``'Const'``).
    score_col : str
        Score column from *classifications* to rank by (default
        ``'pirs_score'``).  Lower PIRS = more constitutive, so the score is
        inverted before computing the curve when *invert_score* is True.
    invert_score : bool
        If ``True`` (default), invert the score so that smaller values rank
        higher (appropriate for PIRS where low = constitutive).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ax(ax)
    merged = classifications[[score_col]].join(
        true_classes[[ground_truth_col]], how='inner'
    ).dropna()
    if merged.empty:
        ax.set_title((title or 'PR curve') + ' (no matched genes)')
        return ax

    scores = merged[score_col].values
    if invert_score:
        scores = -scores
    precision, recall, _ = precision_recall_curve(
        merged[ground_truth_col].values, scores, pos_label=1
    )
    baseline = merged[ground_truth_col].mean()
    ap = np.trapezoid(precision, recall) if hasattr(np, 'trapezoid') else np.trapz(precision, recall)
    ax.plot(recall, precision, color='#4878CF', lw=1.2,
            label=f'{score_col} (AP={ap:.3f})')
    ax.axhline(baseline, color='#CC4444', ls=':', lw=0.9,
               label='random classifier')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title or f'PR curve: {ground_truth_col}')
    ax.legend(frameon=False, fontsize=8)
    sns.despine(ax=ax)
    return ax


def classification_roc(
    classifications,
    true_classes,
    tasks=None,
    ax=None,
    title='ROC: Classifier scores vs ground truth',
):
    """ROC curves for multiple binary classification tasks.

    For each task in *tasks*, the function pairs a ``Classifier`` score column
    against a binary ground-truth column.  This lets you assess, for example,
    whether PIRS score discriminates constitutive genes and whether TauMean
    discriminates circadian genes — on the same axes.

    Parameters
    ----------
    classifications : pd.DataFrame
        Output of ``Classifier.classify()``.
    true_classes : pd.DataFrame
        Binary ground-truth table with columns like ``Const``, ``Circadian``,
        ``Linear``.
    tasks : list of (str, str, bool) or None
        Each entry is ``(score_col, truth_col, invert)``.  ``invert=True``
        means lower score = positive (used for PIRS score and p-values).
        Defaults to sensible combinations based on available columns.
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ax(ax)

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

    palette = sns.color_palette('muted', n_colors=max(len(tasks), 1))
    for (score_col, truth_col, invert), color in zip(tasks, palette):
        merged = classifications[[score_col]].join(
            true_classes[[truth_col]], how='inner'
        ).dropna()
        if merged.empty:
            continue
        scores = -merged[score_col].values if invert else merged[score_col].values
        fpr, tpr, _ = roc_curve(merged[truth_col].values, scores, pos_label=1)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.2,
                label=f'{score_col}→{truth_col}  AUC={auc_val:.3f}')

    ax.plot([0, 1], [0, 1], color='#999999', ls='--', lw=0.8, label='random')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(title)
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    sns.despine(ax=ax)
    return ax
