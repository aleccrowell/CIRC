"""Evaluation metrics for expression classification benchmarking.

These functions compute performance metrics (ROC AUC, average precision)
against simulation ground truth.  They are intentionally decoupled from
matplotlib so they can be used in automated pipelines without a display.

Plotting wrappers that visualise these metrics live in
``circ.visualization.benchmarks``.
"""
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve, auc


# ---------------------------------------------------------------------------
# Legacy-format ROC (merged_dict interface used by circ.limbr.simulations)
# ---------------------------------------------------------------------------

def roc_auc(merged_dict):
    """Compute area under the ROC curve for each method.

    Parameters
    ----------
    merged_dict : dict[str, pd.DataFrame]
        ``{tag: df}`` where each ``df`` has a binary truth column and a score
        column.  The truth column must be named ``truth`` or ``Circadian``, and
        the score column ``score`` or ``GammaBH``.

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
# Classifier-output evaluation (Classifier + simulate ground-truth interface)
# ---------------------------------------------------------------------------

def _default_tasks(classifications, true_classes):
    """Return the default (score_col, truth_col, invert) task list."""
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
    return tasks


def classification_auc(classifications, true_classes, tasks=None):
    """Compute ROC AUC for multiple binary classification tasks.

    Parameters
    ----------
    classifications : pd.DataFrame
        Output of ``Classifier.classify()``, indexed by gene ID.
    true_classes : pd.DataFrame
        Simulation ground-truth table with binary columns such as ``Const``,
        ``Circadian``, and ``Linear``.
    tasks : list of (str, str, bool) or None
        Each entry is ``(score_col, truth_col, invert)``.  ``invert=True``
        means a lower score indicates the positive class (used for PIRS score
        and p-values).  Defaults to sensible combinations based on available
        columns.

    Returns
    -------
    dict[(str, str), float]
        Mapping from ``(score_col, truth_col)`` to AUC value.  Tasks where
        the merged data is empty or has only one class are omitted.
    """
    if tasks is None:
        tasks = _default_tasks(classifications, true_classes)

    result = {}
    for score_col, truth_col, invert in tasks:
        merged = (
            classifications[[score_col]]
            .join(true_classes[[truth_col]], how='inner')
            .dropna()
        )
        if merged.empty or merged[truth_col].nunique() < 2:
            continue
        scores = -merged[score_col].values if invert else merged[score_col].values
        fpr, tpr, _ = roc_curve(merged[truth_col].values, scores, pos_label=1)
        result[(score_col, truth_col)] = auc(fpr, tpr)
    return result


def classification_ap(
    classifications,
    true_classes,
    ground_truth_col='Const',
    score_col='pirs_score',
    invert_score=True,
):
    """Compute average precision (PR AUC) for a single classification task.

    Parameters
    ----------
    classifications : pd.DataFrame
        Output of ``Classifier.classify()``, indexed by gene ID.
    true_classes : pd.DataFrame
        Simulation ground-truth table with binary columns.
    ground_truth_col : str
        Column in *true_classes* to treat as positive (default ``'Const'``).
    score_col : str
        Score column from *classifications* to rank by (default
        ``'pirs_score'``).
    invert_score : bool
        If ``True`` (default), invert the score before computing the curve.

    Returns
    -------
    float or None
        Average precision, or ``None`` if the merged data is empty or
        contains only one class.
    """
    merged = (
        classifications[[score_col]]
        .join(true_classes[[ground_truth_col]], how='inner')
        .dropna()
    )
    if merged.empty or merged[ground_truth_col].nunique() < 2:
        return None
    scores = -merged[score_col].values if invert_score else merged[score_col].values
    ap = average_precision_score(merged[ground_truth_col].values, scores)
    return float(ap)
