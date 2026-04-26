"""Benchmark classifier performance against simulation ground truth.

Use case:
    You have run the classifier on simulated data where the true labels are
    known and want to measure detection accuracy via precision-recall and ROC
    curves.  This is the standard validation workflow before applying the
    classifier to real data.

Run:
    poetry run python examples/03_benchmark_simulation.py

Figures are saved to ./figures/.  Pass --show to display interactively.

Note:
    This example requires ground-truth labels, which are only available for
    simulated data.  For real data, use examples/02_validate_decisions.py
    to assess decision quality without known labels.
"""

import sys
from pathlib import Path
import tempfile, os

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from circ.simulations import simulate
from circ.expression_classification.classify import Classifier
import circ.visualization as viz

try:
    import circ.visualization.interactive as iviz
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Simulate with known ground truth
#
# Use a larger dataset and more permutations for reliable PR/ROC estimates.
# The class balance (pcirc=0.25, plin=0.15) leaves ~60 % constitutive genes,
# which is typical of a housekeeping proteomics experiment.
# ---------------------------------------------------------------------------
print("Simulating data (larger run for reliable benchmark) …")
sim = simulate(
    tpoints=8, nrows=500, nreps=2,
    pcirc=0.25, plin=0.15,
    rseed=0,
)
gene_ids = [f"gene_{i:04d}" for i in range(sim.nrows)]

# Build in-memory expression DataFrame and ground-truth label table
expression = pd.DataFrame(sim.sim, index=gene_ids, columns=sim.cols)
expression.index.name = "#"
true_classes = sim._true_classes_df(index=pd.Index(gene_ids, name="#"))

print(f"  Ground-truth counts: {dict(true_classes.sum())}")

print("Classifying with full permutation testing …")
clf = Classifier(expression, reps=2)
result = clf.run_all(
    slope_pvals=True,
    n_permutations=500,
    n_jobs=1,
)
print(result["label"].value_counts().to_string())
print()

# ---------------------------------------------------------------------------
# 2. Precision-recall curves for each binary classification task
#
# Each plot answers: "how well does score X recover class Y?"
#   Constitutive — low PIRS score should recover Const = 1 genes
#   Circadian    — high TauMean or low emp_p should recover Circadian = 1
#   Linear       — low slope_pval should recover Linear = 1
#
# The dashed baseline shows a random classifier (fraction of positives).
# Average Precision (AP) in the legend summarises the curve as a single number.
# ---------------------------------------------------------------------------
print("Section 2: Precision-recall curves …")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

viz.classification_pr(
    result, true_classes,
    ground_truth_col="Const",
    score_col="pirs_score",
    invert_score=True,
    ax=axes[0],
    title="Constitutive detection (PIRS score)",
)

viz.classification_pr(
    result, true_classes,
    ground_truth_col="Circadian",
    score_col="tau_mean",
    invert_score=False,
    ax=axes[1],
    title="Circadian detection (TauMean)",
)

viz.classification_pr(
    result, true_classes,
    ground_truth_col="Linear",
    score_col="slope_pval",
    invert_score=True,
    ax=axes[2],
    title="Linear detection (slope p-value)",
)

plt.tight_layout()
out = FIGURES / "10_pr_curves.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 3. Multi-task ROC comparison
#
# classification_roc auto-detects available score × truth pairings and plots
# all ROC curves on one axis with AUC labels.  This gives a quick summary of
# which detection tasks are hard vs. easy for this dataset.
#
# Interpretation:
#   AUC ≈ 1.0 — near-perfect separation
#   AUC ≈ 0.5 — no better than random
# ---------------------------------------------------------------------------
print("Section 3: Multi-task ROC …")

fig, ax = plt.subplots(figsize=(7, 6))
viz.classification_roc(result, true_classes, ax=ax, title="ROC: all classification tasks")
plt.tight_layout()
out = FIGURES / "11_roc_multitask.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 4. Comparing multiple score columns for the same task
#
# For circadian detection, both TauMean and GammaBH empirical p-value can
# be used as a ranking score.  pr_curve lets you overlay pre-computed curves
# to see which score recovers the ground truth more precisely.
# ---------------------------------------------------------------------------
print("Section 4: Score column comparison (circadian) …")

from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np

def _pr_df(score_col, truth_col, invert):
    merged = result[[score_col]].join(true_classes[[truth_col]], how="inner").dropna()
    scores = -merged[score_col].values if invert else merged[score_col].values
    prec, rec, thr = precision_recall_curve(merged[truth_col].values, scores, pos_label=1)
    ap = average_precision_score(merged[truth_col].values, scores)
    n = len(prec)
    return pd.DataFrame({
        "precision": prec, "recall": rec,
        "method": [f"{score_col} (AP={ap:.3f})"] * n,
        "rep": [0] * n,
    })

curves = pd.concat([
    _pr_df("tau_mean", "Circadian", invert=False),
    _pr_df("emp_p",    "Circadian", invert=True),
], ignore_index=True)

baseline = true_classes["Circadian"].mean()

fig, ax = plt.subplots(figsize=(6, 5))
viz.pr_curve(curves, baseline=baseline, ax=ax)
ax.set_title("Circadian detection: TauMean vs GammaBH p-value")
plt.tight_layout()
out = FIGURES / "12_pr_score_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 5. Interactive benchmark (Plotly) — hover to inspect threshold points
# ---------------------------------------------------------------------------
if _HAS_PLOTLY:
    print("Interactive benchmark …")

    fig_html = iviz.classification_roc(
        result, true_classes,
        title="Interactive ROC: hover for threshold values",
    )
    out = FIGURES / "13_interactive_roc.html"
    fig_html.write_html(str(out))
    print(f"  saved → {out}")

    fig_html = iviz.classification_pr(
        result, true_classes,
        ground_truth_col="Const",
        score_col="pirs_score",
        invert_score=True,
        title="Interactive PR: constitutive detection",
    )
    out = FIGURES / "14_interactive_pr.html"
    fig_html.write_html(str(out))
    print(f"  saved → {out}")

if "--show" in sys.argv:
    plt.show()

print("\nDone. Figures written to", FIGURES)
