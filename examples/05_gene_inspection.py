"""Inspect individual gene profiles after classification.

Use case:
    You have classified your data and want to move from population-level
    summaries to individual gene profiles.  This example shows how to:

      - Use mean_expression_profiles as context for a label group
      - Pull the top rhythmic and constitutive hits by rank
      - Plot a profile gallery (individual time-series per gene)
      - Annotate those genes in the population decision space
      - Save an interactive HTML for open-ended gene hunting

Run:
    poetry run python examples/05_gene_inspection.py

Swap the simulation block for your own data:
    from circ.io import read_expression
    expression = read_expression("your_normalized_data.parquet")
    clf = Classifier(expression, reps=2)
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
from circ.visualization import LABEL_COLORS
import circ.visualization as viz

try:
    import circ.visualization.interactive as iviz
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Setup — simulate and classify
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

print("Classifying …")
clf = Classifier(expression, reps=2)
result = clf.run_all(
    pvals=True,
    slope_pvals=True,
    n_permutations=200,
    n_jobs=1,
)
print(result["label"].value_counts().to_string(), "\n")

# ---------------------------------------------------------------------------
# 2. Select top hits
# ---------------------------------------------------------------------------
# Top rhythmic: highest TauMean within the rhythmic + noisy_rhythmic labels
rhythmic_genes = result[result["label"].isin(["rhythmic", "noisy_rhythmic"])]
top_rhythmic   = rhythmic_genes.nlargest(9, "tau_mean")

# Top constitutive: lowest PIRS score within the constitutive label
const_genes      = result[result["label"] == "constitutive"]
top_constitutive = const_genes.nsmallest(9, "pirs_score")

print(f"Top 9 rhythmic genes (by TauMean):     {top_rhythmic.index.tolist()}")
print(f"Top 9 constitutive genes (by PIRS):    {top_constitutive.index.tolist()}\n")


# ---------------------------------------------------------------------------
# 3. Context — mean label profiles alongside the gallery
#
# mean_expression_profiles shows what the "average" gene in each label looks
# like, providing biological context before drilling into individual genes.
# ---------------------------------------------------------------------------
print("Section 1: Mean profiles + rhythmic gallery …")

fig = plt.figure(figsize=(20, 8))
gs  = fig.add_gridspec(2, 5, hspace=0.55, wspace=0.4)

# Left column: mean profiles for all labels
ax_mean = fig.add_subplot(gs[:, 0])
viz.mean_expression_profiles(expression, result, ax=ax_mean,
                              title="Mean profile\nby label")

# Right 4 columns (2×4 = 8 panels, but we only have 9 genes; reshape to 3×3)
# Use a nested gridspec for the gallery
gs_gallery = gs[:, 1:].subgridspec(3, 3, hspace=0.6, wspace=0.4)
for idx, gene_id in enumerate(top_rhythmic.index):
    r, c = divmod(idx, 3)
    ax   = fig.add_subplot(gs_gallery[r, c])
    viz.gene_profile(expression, gene_id, result, ax=ax)
    ax.set_title(ax.get_title(), fontsize=7)
    ax.set_xlabel(ax.get_xlabel(), fontsize=7)
    ax.tick_params(labelsize=6)

fig.suptitle("Top rhythmic genes — individual profiles (τ ranked)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = FIGURES / "21_rhythmic_profiles.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 4. Constitutive gene profile gallery
#
# Each panel shows the individual time-series for a top constitutive
# candidate, confirming visually that expression is flat across the day.
# The PIRS score in each title quantifies that flatness.
# ---------------------------------------------------------------------------
print("Section 2: Constitutive profiles …")

fig = plt.figure(figsize=(20, 8))
gs  = fig.add_gridspec(2, 5, hspace=0.55, wspace=0.4)

ax_mean = fig.add_subplot(gs[:, 0])
viz.mean_expression_profiles(expression, result,
                              labels=["constitutive", "variable"],
                              ax=ax_mean,
                              title="Mean profile\n(constitutive / variable)")

gs_gallery = gs[:, 1:].subgridspec(3, 3, hspace=0.6, wspace=0.4)
for idx, gene_id in enumerate(top_constitutive.index):
    r, c = divmod(idx, 3)
    ax   = fig.add_subplot(gs_gallery[r, c])
    viz.gene_profile(expression, gene_id, result, ax=ax)
    ax.set_title(ax.get_title(), fontsize=7)
    ax.set_xlabel(ax.get_xlabel(), fontsize=7)
    ax.tick_params(labelsize=6)

fig.suptitle("Top constitutive gene candidates — individual profiles (PIRS ranked)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = FIGURES / "22_constitutive_profiles.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 5. Expression heatmap — population-level view grouped by label
#
# Complements the individual profile galleries by showing all representative
# genes in a single heatmap.  Within each label group genes are sorted by
# hierarchical clustering on their z-scored expression patterns.
# ---------------------------------------------------------------------------
print("Section 3: Expression heatmap …")

fig, ax = plt.subplots(figsize=(8, 9))
viz.expression_heatmap(
    expression, result,
    n_per_label=12,
    ax=ax,
    title="Clustered expression by label",
)
fig.tight_layout()
out = FIGURES / "23_expression_heatmap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 6. Annotated decision space
#
# Overlays gene-ID labels for the top hits on the pirs_vs_tau plot, so you
# can see exactly where each candidate sits relative to the decision boundaries.
# Only annotate a small set to keep the plot readable.
# ---------------------------------------------------------------------------
print("Section 3: Annotated decision space …")

n_label = 6
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, genes, title_suffix, color_key in [
    (axes[0], top_rhythmic.head(n_label),   "rhythmic hits",     "rhythmic"),
    (axes[1], top_constitutive.head(n_label), "constitutive hits", "constitutive"),
]:
    viz.pirs_vs_tau(result, ax=ax, title=f"Decision space — {title_suffix} annotated")
    for gene_id, row in genes.iterrows():
        ax.annotate(
            gene_id,
            (row["pirs_score"], row["tau_mean"]),
            fontsize=6,
            color=LABEL_COLORS[color_key],
            xytext=(4, 4), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color=LABEL_COLORS[color_key],
                            lw=0.6, alpha=0.7),
        )
    # Highlight the annotated genes with larger markers
    ax.scatter(
        genes["pirs_score"], genes["tau_mean"],
        s=60, color=LABEL_COLORS[color_key],
        edgecolors="white", linewidths=0.6, zorder=5,
    )

plt.tight_layout()
out = FIGURES / "23_annotated_decision_space.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 6. Interactive gene hunt
#
# The interactive scatter lets you hover over any gene to see its ID and
# scores.  Save to HTML and open in a browser to explore freely.
# ---------------------------------------------------------------------------
if _HAS_PLOTLY:
    print("Section 4: Interactive gene hunt …")

    fig_html = iviz.pirs_vs_tau(result,
                                title="Hover to identify genes — PIRS vs TauMean")
    out = FIGURES / "24_interactive_gene_hunt.html"
    fig_html.write_html(str(out))
    print(f"  saved → {out}")

    fig_html = iviz.tau_pval_scatter(result,
                                     title="Hover to identify genes — rhythmicity decision space")
    out = FIGURES / "25_interactive_rhythm_space.html"
    fig_html.write_html(str(out))
    print(f"  saved → {out}")

if "--show" in sys.argv:
    plt.show()

print("\nDone. Figures written to", FIGURES)
