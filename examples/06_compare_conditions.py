"""Compare classification results across two experimental conditions.

Use case:
    You have run the classifier on two datasets — for example, wild-type vs
    knockout, or two tissue types — and want to understand what changed:
    which label categories shifted, which rhythmic genes gained or lost
    rhythmicity, whether the same genes serve as stable housekeepers in both
    conditions, and which changes are statistically significant.

How it works:
    ``circ.compare.compare_conditions`` joins the two result DataFrames on
    shared gene IDs and computes:

    * Effect sizes: Δ TauMean, Δ PIRS score, circular Δ phase (±12 h)
    * Rhythmicity status per gene: gained / lost / maintained / never rhythmic
    * Statistical tests (when BooteJTK uncertainty columns are present):
        - Welch's t-test on TauMean using the bootstrap SD from each condition
        - z-test on the circular phase shift for genes rhythmic in both
      Both tests apply Benjamini–Hochberg FDR correction across all genes.

Run:
    poetry run python examples/06_compare_conditions.py

Figures are saved to ./figures/.  Pass --show to display interactively.

Swap the simulation block for real data:
    from circ.io import read_expression
    expr_A = read_expression("condition_A.parquet")
    expr_B = read_expression("condition_B.parquet")
    # Gene IDs must match across the two files for overlap analyses.
"""

import sys
from pathlib import Path

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from circ.simulations import simulate
from circ.expression_classification.classify import Classifier
from circ.visualization import LABEL_COLORS
import circ.visualization as viz
from circ.compare import compare_conditions, label_change_table

try:
    import circ.visualization.interactive as iviz
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

COND_LABELS = ("Condition A (reference)", "Condition B (phase-shifted)")
COND_COLORS = ("#4878CF", "#D65F5F")

# ---------------------------------------------------------------------------
# 1. Setup — two simulations sharing the same gene IDs
#
# The simulation is split into two parts so that the two conditions share a
# realistic set of rhythmic genes:
#
#   Shared core (gene_0000–gene_0059): 60 genes simulated as fully rhythmic
#     in both conditions but with independent random phases, mimicking genes
#     that oscillate in both conditions but peak at different times.
#
#   Condition-specific background (gene_0060–gene_0299):
#     Condition A has additional rhythmic and linear genes; Condition B has
#     fewer rhythmic genes and more linear ones (clock-disrupted background).
#
# This structure ensures that the phase shift histogram (genes rhythmic in
# BOTH conditions) contains real data.  With purely independent simulations
# the rhythmic gene sets rarely overlap at typical detection thresholds.
# ---------------------------------------------------------------------------
print("Simulating shared rhythmic core …")
sim_core_A = simulate(tpoints=8, nrows=60, nreps=2, pcirc=1.0, plin=0.0, rseed=10, amp_noise=0.35)
sim_core_B = simulate(tpoints=8, nrows=60, nreps=2, pcirc=1.0, plin=0.0, rseed=11, amp_noise=0.35)

print("Simulating condition-specific backgrounds …")
sim_bg_A = simulate(tpoints=8, nrows=240, nreps=2, pcirc=0.20, plin=0.10, rseed=1)
sim_bg_B = simulate(tpoints=8, nrows=240, nreps=2, pcirc=0.05, plin=0.25, rseed=2)

gene_ids = [f"gene_{i:04d}" for i in range(300)]
cols = sim_core_A.cols

expr_A = pd.DataFrame(
    np.vstack([sim_core_A.sim, sim_bg_A.sim]),
    index=gene_ids, columns=cols,
)
expr_A.index.name = "#"

expr_B = pd.DataFrame(
    np.vstack([sim_core_B.sim, sim_bg_B.sim]),
    index=gene_ids, columns=cols,
)
expr_B.index.name = "#"

# run_all() includes tau_std, phase_std, and n_boots — the bootstrap uncertainty
# estimates from BooteJTK that compare_conditions() needs for statistical tests.
print("Classifying condition A …")
clf_A = Classifier(expr_A, reps=2)
result_A = clf_A.run_all(pvals=True, slope_pvals=True, n_permutations=200, n_jobs=1)

print("Classifying condition B …")
clf_B = Classifier(expr_B, reps=2)
result_B = clf_B.run_all(pvals=True, slope_pvals=True, n_permutations=200, n_jobs=1)

print("\nCondition A labels:")
print(result_A["label"].value_counts().to_string())
print("\nCondition B labels:")
print(result_B["label"].value_counts().to_string(), "\n")

# ---------------------------------------------------------------------------
# 2. Label distribution comparison
#
# Side-by-side bar charts reveal which label categories expanded or shrank.
# A shared x-axis makes gene counts visually comparable between conditions.
# ---------------------------------------------------------------------------
print("Section 1: Label distribution comparison …")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
viz.label_distribution(result_A, ax=axes[0], title=COND_LABELS[0])
viz.label_distribution(result_B, ax=axes[1], title=COND_LABELS[1])

xmax = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
axes[0].set_xlim(0, xmax)
axes[1].set_xlim(0, xmax)

plt.tight_layout()
out = FIGURES / "26_label_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 3. Phase distribution comparison
#
# Overlaying the phase wheels shows whether the timing of peak expression
# shifted.  If the KO delays or advances rhythmic gene phases, the angular
# distribution will rotate.
# ---------------------------------------------------------------------------
print("Section 2: Phase wheel comparison …")

fig = plt.figure(figsize=(10, 5))
ax_A = fig.add_subplot(1, 2, 1, projection="polar")
ax_B = fig.add_subplot(1, 2, 2, projection="polar")

viz.phase_wheel(result_A, ax=ax_A, title=f"Phase — {COND_LABELS[0]}")
viz.phase_wheel(result_B, ax=ax_B, title=f"Phase — {COND_LABELS[1]}")

plt.tight_layout()
out = FIGURES / "27_phase_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 4. Constitutive gene overlap
#
# Genes constitutive in BOTH conditions are the most reliable reference
# gene candidates — their stability is not condition-specific.
# ---------------------------------------------------------------------------
print("Section 3: Constitutive gene overlap …")

const_A = set(result_A[result_A["label"] == "constitutive"].index)
const_B = set(result_B[result_B["label"] == "constitutive"].index)
both    = const_A & const_B
only_A  = const_A - const_B
only_B  = const_B - const_A

print(f"  Constitutive in A only : {len(only_A)}")
print(f"  Constitutive in both   : {len(both)}")
print(f"  Constitutive in B only : {len(only_B)}\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
viz.top_constitutive_candidates(result_A, n_top=15, ax=axes[0],
                                title=f"Top candidates — {COND_LABELS[0]}")
viz.top_constitutive_candidates(result_B, n_top=15, ax=axes[1],
                                title=f"Top candidates — {COND_LABELS[1]}")

top15_A = set(result_A.dropna(subset=["pirs_score"]).nsmallest(15, "pirs_score").index)
top15_B = set(result_B.dropna(subset=["pirs_score"]).nsmallest(15, "pirs_score").index)
shared_top = top15_A & top15_B
if shared_top:
    for ax in axes:
        for tick in ax.get_yticklabels():
            if tick.get_text() in shared_top:
                tick.set_fontweight("bold")
                tick.set_color("#2E8B57")

shared_patch = mpatches.Patch(color="#2E8B57",
                              label=f"In both top-15 ({len(shared_top)} genes)")
fig.legend(handles=[shared_patch], loc="lower center", ncol=1,
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
out = FIGURES / "28_constitutive_overlap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 5. Statistical comparison — compare_conditions()
#
# Joins the two result DataFrames on shared gene IDs and returns:
#   - Effect sizes: delta_tau, delta_pirs, delta_phase
#   - Rhythmicity status per gene (gained / lost / maintained / never)
#   - tau_pval / tau_padj: Welch t-test on bootstrap TauMean distributions
#   - phase_pval / phase_padj: z-test on circular phase shift
# ---------------------------------------------------------------------------
print("Section 4: Statistical comparison …")

comparison = compare_conditions(result_A, result_B)

print("\nRhythmicity status breakdown:")
print(comparison["rhythmicity_status"].value_counts().to_string())

sig_tau   = comparison["tau_padj"].lt(0.05).sum()
sig_phase = comparison["phase_padj"].lt(0.05).sum() if "phase_padj" in comparison.columns else 0
print(f"\nSignificant tau changes  (tau_padj < 0.05) : {sig_tau}")
print(f"Significant phase shifts (phase_padj < 0.05): {sig_phase}")

print("\nLabel transition table (A → B):")
print(label_change_table(comparison).to_string(), "\n")

# comparison_summary auto-selects panels from: rhythmicity shift scatter,
# label transition heatmap, phase shift histogram, and delta tau volcano.
fig = viz.comparison_summary(
    comparison,
    outpath=str(FIGURES / "29_comparison_summary.png"),
)
plt.close(fig)
print(f"  saved → {FIGURES / '29_comparison_summary.png'}")

# ---------------------------------------------------------------------------
# 6. Top genes with significant rhythmicity or phase changes
# ---------------------------------------------------------------------------
print("Section 5: Top significant genes …")

sig = comparison[comparison["tau_padj"] < 0.05]
lost = sig[sig["rhythmicity_status"] == "lost"].nsmallest(5, "delta_tau")
gained = sig[sig["rhythmicity_status"] == "gained"].nlargest(5, "delta_tau")

if not lost.empty:
    print("\nTop genes losing rhythmicity (tau_padj < 0.05):")
    print(lost[["tau_mean_A", "tau_mean_B", "delta_tau", "tau_padj"]].to_string())

if not gained.empty:
    print("\nTop genes gaining rhythmicity (tau_padj < 0.05):")
    print(gained[["tau_mean_A", "tau_mean_B", "delta_tau", "tau_padj"]].to_string())

if "phase_padj" in comparison.columns:
    sig_phase_df = comparison[
        comparison["phase_padj"].lt(0.05) &
        comparison["rhythmicity_status"].eq("maintained_rhythmic")
    ].copy()
    if not sig_phase_df.empty:
        sig_phase_df["abs_shift"] = sig_phase_df["delta_phase"].abs()
        print("\nTop genes with significant phase shifts (phase_padj < 0.05):")
        print(sig_phase_df.nlargest(5, "abs_shift")[
            ["phase_A", "phase_B", "delta_phase", "phase_padj"]
        ].to_string())

# ---------------------------------------------------------------------------
# 7. Interactive condition comparison
# ---------------------------------------------------------------------------
if _HAS_PLOTLY:
    print("\nSection 6: Interactive figure …")

    hover_texts = [
        f"<b>{g}</b><br>"
        f"TauMean A: {row['tau_mean_A']:.3f}<br>"
        f"TauMean B: {row['tau_mean_B']:.3f}<br>"
        f"Δ tau: {row['delta_tau']:+.3f}<br>"
        f"{row['rhythmicity_status']}"
        for g, row in comparison.iterrows()
    ]

    import plotly.graph_objects as go

    _STATUS_COLORS = {
        'maintained_rhythmic':    '#6ACC65',
        'gained':                 '#D65F5F',
        'lost':                   '#4878CF',
        'maintained_nonrhythmic': '#CCCCCC',
    }
    traces = []
    lim = max(comparison["tau_mean_A"].max(), comparison["tau_mean_B"].max()) * 1.05
    for status, color in _STATUS_COLORS.items():
        sub  = comparison[comparison["rhythmicity_status"] == status]
        htxt = [hover_texts[comparison.index.get_loc(g)] for g in sub.index]
        if sub.empty:
            continue
        traces.append(go.Scatter(
            x=sub["tau_mean_A"], y=sub["tau_mean_B"],
            mode="markers",
            name=f"{status} (n={len(sub)})",
            marker=dict(color=color, size=5, opacity=0.7),
            hovertext=htxt,
            hovertemplate="%{hovertext}<extra></extra>",
        ))
    traces.append(go.Scatter(
        x=[0, lim], y=[0, lim], mode="lines",
        line=dict(color="#999999", dash="dash", width=1),
        name="y = x", hoverinfo="skip",
    ))
    fig_html = go.Figure(traces)
    fig_html.update_layout(
        title="Rhythmicity shift — hover to identify genes",
        xaxis_title=f"TauMean — {COND_LABELS[0]}",
        yaxis_title=f"TauMean — {COND_LABELS[1]}",
    )
    out = FIGURES / "30_interactive_rhythmicity_shift.html"
    fig_html.write_html(str(out))
    print(f"  saved → {out}")

if "--show" in sys.argv:
    plt.show()

print("\nDone. Figures written to", FIGURES)
