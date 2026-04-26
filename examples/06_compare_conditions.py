"""Compare classification results across two experimental conditions.

Use case:
    You have run the classifier on two datasets — for example, wild-type vs
    knockout, or two tissue types — and want to understand what changed:
    which label categories shifted, which rhythmic genes gained or lost
    rhythmicity, and whether the same genes serve as stable housekeepers
    in both conditions.

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

try:
    import circ.visualization.interactive as iviz
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

COND_LABELS = ("Condition A (WT-like)", "Condition B (KO-like)")
COND_COLORS = ("#4878CF", "#D65F5F")   # blue / red

# ---------------------------------------------------------------------------
# 1. Setup — two simulations sharing the same gene IDs
#
# Condition A: higher circadian fraction (more rhythmic genes)
# Condition B: lower circadian fraction, more linear (e.g., clock-disrupted)
#
# Both use the same gene_ids so downstream overlap analyses are meaningful.
# ---------------------------------------------------------------------------
print("Simulating condition A …")
sim_A = simulate(tpoints=8, nrows=300, nreps=2, pcirc=0.30, plin=0.10, rseed=1)
print("Simulating condition B …")
sim_B = simulate(tpoints=8, nrows=300, nreps=2, pcirc=0.10, plin=0.25, rseed=2)

gene_ids = [f"gene_{i:04d}" for i in range(sim_A.nrows)]

def _make_expr(sim):
    df = pd.DataFrame(sim.sim, index=gene_ids, columns=sim.cols)
    df.index.name = "#"
    return df

expr_A = _make_expr(sim_A)
expr_B = _make_expr(sim_B)

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
# Side-by-side bar charts reveal which label categories expanded or shrank
# between conditions.  In a clock-disrupted condition you expect fewer
# rhythmic genes and potentially more variable or linear ones.
# ---------------------------------------------------------------------------
print("Section 1: Label distribution comparison …")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
viz.label_distribution(result_A, ax=axes[0], title=COND_LABELS[0])
viz.label_distribution(result_B, ax=axes[1], title=COND_LABELS[1])

# Shared x-axis range so gene counts are visually comparable across conditions
xmax = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
axes[0].set_xlim(0, xmax)
axes[1].set_xlim(0, xmax)

plt.tight_layout()
out = FIGURES / "20_label_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 3. Phase distribution comparison
#
# Overlaying the phase wheels from both conditions shows whether the timing
# of peak expression shifted.  If the KO delays or advances the phase of
# rhythmic genes, you will see the angular distribution rotate.
# ---------------------------------------------------------------------------
print("Section 2: Phase wheel comparison …")

fig = plt.figure(figsize=(10, 5))
ax_A = fig.add_subplot(1, 2, 1, projection="polar")
ax_B = fig.add_subplot(1, 2, 2, projection="polar")

viz.phase_wheel(result_A, ax=ax_A, title=f"Phase — {COND_LABELS[0]}")
viz.phase_wheel(result_B, ax=ax_B, title=f"Phase — {COND_LABELS[1]}")

plt.tight_layout()
out = FIGURES / "21_phase_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 4. Constitutive gene overlap
#
# Genes that are constitutively expressed in BOTH conditions are the most
# reliable internal reference gene candidates — their stability is not
# condition-specific.  Genes constitutive in only one condition should be
# used with caution.
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

# Side-by-side top_constitutive_candidates; highlight shared genes
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
viz.top_constitutive_candidates(result_A, n_top=15, ax=axes[0],
                                title=f"Top candidates — {COND_LABELS[0]}")
viz.top_constitutive_candidates(result_B, n_top=15, ax=axes[1],
                                title=f"Top candidates — {COND_LABELS[1]}")

# Mark genes that appear in both conditions' top-15 lists
top15_A = set(result_A.dropna(subset=["pirs_score"]).nsmallest(15, "pirs_score").index)
top15_B = set(result_B.dropna(subset=["pirs_score"]).nsmallest(15, "pirs_score").index)
shared_top = top15_A & top15_B
if shared_top:
    for ax in axes:
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        for i, lbl in enumerate(ytick_labels):
            if lbl in shared_top:
                ax.get_yticklabels()[i].set_fontweight("bold")
                ax.get_yticklabels()[i].set_color("#2E8B57")

shared_patch = mpatches.Patch(color="#2E8B57",
                              label=f"In both top-15 ({len(shared_top)} genes)")
fig.legend(handles=[shared_patch], loc="lower center", ncol=1,
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
out = FIGURES / "22_constitutive_overlap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 5. Rhythmicity scatter — TauMean A vs TauMean B
#
# Each point is one gene.  Points above the diagonal gained rhythmicity in
# condition B; points below lost it.  Color encodes whether the gene was
# called rhythmic in both, one, or neither condition.
# ---------------------------------------------------------------------------
print("Section 4: Rhythmicity shift scatter …")

# Rhythmic = in (rhythmic | noisy_rhythmic) label
rhythmic_A = set(result_A[result_A["label"].isin(["rhythmic", "noisy_rhythmic"])].index)
rhythmic_B = set(result_B[result_B["label"].isin(["rhythmic", "noisy_rhythmic"])].index)

tau = result_A[["tau_mean"]].join(result_B[["tau_mean"]], lsuffix="_A", rsuffix="_B",
                                  how="inner").dropna()

def _rhythm_category(gene_id):
    in_A = gene_id in rhythmic_A
    in_B = gene_id in rhythmic_B
    if in_A and in_B:   return "rhythmic in both",  "#6ACC65"
    if in_A:            return "only in A",          COND_COLORS[0]
    if in_B:            return "only in B",          COND_COLORS[1]
    return                     "not rhythmic",       "#CCCCCC"

tau["category"], tau["color"] = zip(*[_rhythm_category(g) for g in tau.index])

fig, ax = plt.subplots(figsize=(6, 6))
for cat, color in [
    ("not rhythmic",   "#CCCCCC"),
    ("only in A",      COND_COLORS[0]),
    ("only in B",      COND_COLORS[1]),
    ("rhythmic in both", "#6ACC65"),
]:
    sub = tau[tau["category"] == cat]
    if sub.empty:
        continue
    ax.scatter(sub["tau_mean_A"], sub["tau_mean_B"],
               color=color, s=10, alpha=0.6, label=f"{cat} (n={len(sub)})",
               rasterized=len(tau) > 500)

lim = max(tau["tau_mean_A"].max(), tau["tau_mean_B"].max()) * 1.05
ax.plot([0, lim], [0, lim], color="#999999", ls="--", lw=0.8, alpha=0.7)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel(f"TauMean — {COND_LABELS[0]}")
ax.set_ylabel(f"TauMean — {COND_LABELS[1]}")
ax.set_title("Rhythmicity shift: TauMean per condition")
ax.legend(loc="upper left", frameon=False, fontsize=8, markerscale=1.5)
sns.despine(ax=ax)

plt.tight_layout()
out = FIGURES / "23_rhythmicity_shift.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 6. Interactive condition comparison
# ---------------------------------------------------------------------------
if _HAS_PLOTLY:
    print("Section 5: Interactive figures …")

    # Hover to identify which genes changed rhythmicity
    hover_texts = []
    for gene_id, row in tau.iterrows():
        txt = (f"<b>{gene_id}</b><br>"
               f"TauMean A: {row['tau_mean_A']:.3f}<br>"
               f"TauMean B: {row['tau_mean_B']:.3f}<br>"
               f"{row['category']}")
        hover_texts.append(txt)

    import plotly.graph_objects as go
    traces = []
    for cat, color in [
        ("not rhythmic",    "#CCCCCC"),
        ("only in A",       COND_COLORS[0]),
        ("only in B",       COND_COLORS[1]),
        ("rhythmic in both","#6ACC65"),
    ]:
        sub  = tau[tau["category"] == cat]
        htxt = [hover_texts[tau.index.get_loc(g)] for g in sub.index]
        if sub.empty:
            continue
        traces.append(go.Scatter(
            x=sub["tau_mean_A"], y=sub["tau_mean_B"],
            mode="markers",
            name=f"{cat} (n={len(sub)})",
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
    out = FIGURES / "24_interactive_rhythmicity_shift.html"
    fig_html.write_html(str(out))
    print(f"  saved → {out}")

if "--show" in sys.argv:
    plt.show()

print("\nDone. Figures written to", FIGURES)
