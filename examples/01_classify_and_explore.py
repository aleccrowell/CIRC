"""Classify and explore results by use-case group.

Use case:
    You have classified your expression data and want to explore the results
    through three focused lenses: dataset overview, rhythmicity deep-dive,
    and constitutive gene characterization.

Run:
    poetry run python examples/01_classify_and_explore.py

Figures are saved to ./figures/.  Pass --show to display interactively instead.

Swap the simulation block for your own data:
    expression = circ.io.read_expression("your_data.parquet")
    clf = Classifier(expression, reps=<your_replicate_count>)
"""

import sys
from pathlib import Path

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from circ.simulations import simulate
from circ.expression_classification.classify import Classifier
import circ.visualization as viz

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Setup — simulate and classify
# ---------------------------------------------------------------------------
# For your own data, replace this block with:
#   from circ.io import read_expression
#   expression = read_expression("your_normalized_data.parquet")
#   clf = Classifier(expression, reps=2)

print("Simulating data …")
sim = simulate(
    tpoints=8, nrows=300, nreps=2,
    pcirc=0.25, plin=0.15,
    rseed=42,
)
gene_ids  = [f"gene_{i:04d}" for i in range(sim.nrows)]
expression = pd.DataFrame(sim.sim, index=gene_ids, columns=sim.cols)
expression.index.name = "#"

print("Classifying …")
clf = Classifier(expression, reps=2)
result = clf.run_all(
    pvals=True,
    slope_pvals=True,
    n_permutations=200,
    n_jobs=1,
)
print(result["label"].value_counts().to_string())
print()

# ---------------------------------------------------------------------------
# 2. Overview — what did my experiment produce?
#
# These three plots answer the first question after any classification run:
# how are my genes distributed, how do the PIRS scores separate, and does
# the mean time-series profile for each label look biologically sensible?
# ---------------------------------------------------------------------------
print("Group 1: Overview …")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

viz.label_distribution(result, ax=axes[0])

viz.pirs_score_distribution(result, ax=axes[1])

# mean_expression_profiles requires the raw expression data alongside results
viz.mean_expression_profiles(expression, result, ax=axes[2])

plt.tight_layout()
out = FIGURES / "01_overview.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 3. Rhythmicity — when and how strongly do rhythmic genes oscillate?
#
# tau_pval_scatter shows the BooteJTK decision space (which genes cleared
# both the tau and FDR thresholds).  phase_wheel and period_distribution
# summarise timing across the rhythmic population.  phase_amplitude_scatter
# lets you find genes with the strongest rhythms at each phase.
# ---------------------------------------------------------------------------
print("Group 2: Rhythmicity …")

fig = plt.figure(figsize=(20, 5))
ax0 = fig.add_subplot(1, 4, 1)
ax1 = fig.add_subplot(1, 4, 2, projection="polar")
ax2 = fig.add_subplot(1, 4, 3)
ax3 = fig.add_subplot(1, 4, 4)

viz.tau_pval_scatter(result, ax=ax0)
viz.phase_wheel(result, ax=ax1)
viz.period_distribution(result, ax=ax2)
viz.phase_amplitude_scatter(result, ax=ax3)

plt.tight_layout()
out = FIGURES / "02_rhythmicity.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 4. Constitutive genes — identifying stable reference candidates
#
# pirs_vs_tau shows the joint PIRS × tau decision space with threshold lines.
# pirs_pval_scatter confirms which constitutive candidates are statistically
# supported.  top_constitutive_candidates ranks the best reference gene picks.
# threshold_sensitivity (ECDF) shows how well the labels separate at the
# current PIRS cutoff — ideal if the constitutive ECDF rises sharply to the
# left of the cut while other labels sit to the right.
# ---------------------------------------------------------------------------
print("Group 3: Constitutive characterisation …")

fig, axes = plt.subplots(1, 4, figsize=(20, 4))

viz.pirs_vs_tau(result, ax=axes[0])

viz.pirs_pval_scatter(result, ax=axes[1])

viz.top_constitutive_candidates(result, n_top=20, ax=axes[2])

viz.threshold_sensitivity(result, ax=axes[3])

plt.tight_layout()
out = FIGURES / "03_constitutive.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# Bonus: all-in-one adaptive summary
# ---------------------------------------------------------------------------
print("Adaptive summary …")
fig = viz.classification_summary(
    result,
    outpath=str(FIGURES / "04_summary.png"),
)
plt.close(fig)
print(f"  saved → {FIGURES / '04_summary.png'}")

if "--show" in sys.argv:
    plt.show()

print("\nDone. Figures written to", FIGURES)
