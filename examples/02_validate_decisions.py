"""Validate classification decisions and tune thresholds.

Use case:
    You want to understand *why* the classifier assigned each label, examine
    the joint significance decision space, and explore how changing the
    pirs_percentile or tau_threshold affects the label distribution before
    committing to a final setting.

Run:
    poetry run python examples/02_validate_decisions.py

Figures are saved to ./figures/.  Pass --show to display interactively.

Interactive HTML files are also written so you can hover over individual
genes in a browser:
    figures/07_interactive_candidates.html
    figures/08_interactive_phase.html
"""

import sys
from pathlib import Path

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
    print("plotly not installed — interactive HTML figures will be skipped.")
    print("Install with: poetry install --extras interactive\n")

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Setup — simulate and run the full classifier pipeline once
#
# The expensive steps (PIRS permutations, BooteJTK) run only once here.
# We call clf.classify() again later with different thresholds — it's fast
# because it re-uses the already-computed scores.
# ---------------------------------------------------------------------------
print("Simulating data …")
sim = simulate(
    tpoints=8, nrows=300, nreps=2,
    pcirc=0.25, plin=0.15,
    rseed=42,
)
gene_ids   = [f"gene_{i:04d}" for i in range(sim.nrows)]
expression = pd.DataFrame(sim.sim, index=gene_ids, columns=sim.cols)
expression.index.name = "#"

print("Running PIRS + BooteJTK (permutations run once) …")
clf = Classifier(expression, reps=2)
clf.run_pirs(pvals=True, slope_pvals=True, n_permutations=200, n_jobs=1)
clf.run_bootjtk()
result = clf.classify(pirs_percentile=50)
print(result["label"].value_counts().to_string())
print()

# ---------------------------------------------------------------------------
# 2. Decision-space validation
#
# These plots reveal the multi-axis decision logic:
#   volcano        — PIRS score vs rhythm significance (4-quadrant view)
#   slope_vs_rhythm — separates rhythmic from linear from constitutive/variable
#   slope_pval_scatter — which genes have a significant linear trend
#
# Genes that sit near a decision boundary deserve extra scrutiny.
# ---------------------------------------------------------------------------
print("Group 4: Decision-space validation …")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

viz.volcano(result, ax=axes[0])

viz.slope_vs_rhythm(result, ax=axes[1])

viz.slope_pval_scatter(result, ax=axes[2])

plt.tight_layout()
out = FIGURES / "05_decision_space.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 3. Threshold sensitivity — choose your pirs_percentile
#
# clf.classify() re-runs instantly since scores are already computed.
# Compare three settings side-by-side:
#   p25 → more selective constitutive call (fewer but higher-confidence genes)
#   p50 → default balanced split
#   p75 → permissive constitutive call (more genes, but lower-scoring ones)
#
# The ECDF plots show whether the labels separate cleanly at each cutoff.
# The classification_summary panels let you see the downstream knock-on.
# ---------------------------------------------------------------------------
print("Group 4b: Threshold sensitivity …")

percentiles = [25, 50, 75]
results = {
    p: clf.classify(pirs_percentile=p)
    for p in percentiles
}

# --- ECDF comparison: one threshold_sensitivity per column ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (p, res) in zip(axes, results.items()):
    viz.threshold_sensitivity(res, pirs_percentile=p, ax=ax,
                              title=f"ECDF — PIRS p{p}")
plt.tight_layout()
out = FIGURES / "06_threshold_ecdf.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# --- Label counts across the three settings ---
print("\n  Label counts by pirs_percentile:")
counts = pd.DataFrame(
    {f"p{p}": res["label"].value_counts() for p, res in results.items()}
).fillna(0).astype(int)
print(counts.to_string())
print()

# ---------------------------------------------------------------------------
# 4. Interactive exploration — gene-level inspection in the browser
#
# The interactive module returns Plotly figures.  Hovering over a point
# shows the gene ID and all associated scores, making it easy to flag
# specific genes for follow-up (e.g. known reference genes, clock genes).
# ---------------------------------------------------------------------------
if _HAS_PLOTLY:
    print("Interactive figures …")

    fig_html = iviz.top_constitutive_candidates(result, n_top=30)
    out = FIGURES / "07_interactive_candidates.html"
    fig_html.write_html(str(out))
    print(f"  saved → {out}")

    fig_html = iviz.phase_amplitude_scatter(result)
    out = FIGURES / "08_interactive_phase.html"
    fig_html.write_html(str(out))
    print(f"  saved → {out}")

    fig_html = iviz.volcano(result)
    out = FIGURES / "09_interactive_volcano.html"
    fig_html.write_html(str(out))
    print(f"  saved → {out}")

if "--show" in sys.argv:
    plt.show()

print("\nDone. Figures written to", FIGURES)
