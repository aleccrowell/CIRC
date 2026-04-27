"""Compare circadian classifications from proteomics and gene-expression experiments.

Use case:
    You have circadian profiling data from the same biological system measured
    at two molecular layers — mass-spectrometry proteomics and RNA-seq — and
    want to identify which proteins/genes are rhythmic at each level, whether
    oscillation phase is conserved across layers, and where the layers diverge
    (e.g. post-transcriptionally regulated rhythms).

How it works:
    Both datasets are classified independently with ``Classifier``.  Because
    the proteomics Protein identifiers match the RNA-seq gene identifiers,
    ``compare_conditions`` finds the shared features automatically and computes
    per-feature effect sizes and significance tests.

    If your proteomics result is at *peptide* level (a ``(Peptide, Protein)``
    MultiIndex), call ``aggregate_to_protein`` first — this is demonstrated
    in the final section.

Run:
    poetry run python examples/07_compare_proteomics_vs_expression.py

Figures are saved to ./figures/.  Pass --show to display interactively.

Swap the simulation block for real data:
    from circ.io import read_expression
    # Protein-level proteomics (e.g. after circ.limbr SVA normalization):
    prot_expr = read_expression("proteomics_protein_level.parquet")
    # RNA-seq:
    rna_expr  = read_expression("rnaseq.parquet")
    # Protein IDs in prot_expr.index must match gene IDs in rna_expr.index.
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
from circ.compare import aggregate_to_protein, compare_conditions, label_change_table

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

LAYER_LABELS = ("Proteomics", "RNA-seq")

# ---------------------------------------------------------------------------
# 1. Simulate paired multi-omic datasets
#
# 200 features share the same identifiers (protein ID = gene ID) across both
# molecular layers.  Three tiers create realistic cross-layer divergence:
#
#   feat_0000–0039  Shared rhythmic (40):  oscillate at both protein and mRNA
#     level — transcriptionally driven rhythms.
#   feat_0040–0059  Proteomics-only rhythmic (20):  rhythmic at protein level
#     but flat mRNA — e.g. post-translational regulation or slow turnover.
#   feat_0060–0079  RNA-only rhythmic (20):  rhythmic mRNA but non-rhythmic
#     protein — e.g. rapid protein degradation buffers the oscillation.
#   feat_0080–0199  Non-rhythmic background (120).
# ---------------------------------------------------------------------------
print("Simulating multi-omic datasets …")

sim_core_prot      = simulate(tpoints=8, nrows=40,  nreps=2, pcirc=1.0, plin=0.0, rseed=20, amp_noise=0.35)
sim_core_rna       = simulate(tpoints=8, nrows=40,  nreps=2, pcirc=1.0, plin=0.0, rseed=21, amp_noise=0.35)
sim_prot_only_rhy  = simulate(tpoints=8, nrows=20,  nreps=2, pcirc=1.0, plin=0.0, rseed=22, amp_noise=0.35)
sim_prot_only_flat = simulate(tpoints=8, nrows=20,  nreps=2, pcirc=0.0, plin=0.0, rseed=26)
sim_rna_only_flat  = simulate(tpoints=8, nrows=20,  nreps=2, pcirc=0.0, plin=0.0, rseed=27)
sim_rna_only_rhy   = simulate(tpoints=8, nrows=20,  nreps=2, pcirc=1.0, plin=0.0, rseed=23, amp_noise=0.35)
sim_bg_prot        = simulate(tpoints=8, nrows=120, nreps=2, pcirc=0.0, plin=0.05, rseed=24)
sim_bg_rna         = simulate(tpoints=8, nrows=120, nreps=2, pcirc=0.0, plin=0.05, rseed=25)

feature_ids = [f"feat_{i:04d}" for i in range(200)]
cols = sim_core_prot.cols

prot_expr = pd.DataFrame(
    np.vstack([
        sim_core_prot.sim,       # feat_0000–0039: rhythmic in both
        sim_prot_only_rhy.sim,   # feat_0040–0059: prot-only rhythmic
        sim_rna_only_flat.sim,   # feat_0060–0079: non-rhythmic at protein level
        sim_bg_prot.sim,         # feat_0080–0199: background
    ]),
    index=feature_ids, columns=cols,
)
prot_expr.index.name = "#"

rna_expr = pd.DataFrame(
    np.vstack([
        sim_core_rna.sim,        # feat_0000–0039: rhythmic in both
        sim_prot_only_flat.sim,  # feat_0040–0059: non-rhythmic in RNA
        sim_rna_only_rhy.sim,    # feat_0060–0079: rna-only rhythmic
        sim_bg_rna.sim,          # feat_0080–0199: background
    ]),
    index=feature_ids, columns=cols,
)
rna_expr.index.name = "#"

# ---------------------------------------------------------------------------
# 2. Classify each molecular layer independently
# ---------------------------------------------------------------------------
print("Classifying proteomics layer …")
clf_prot    = Classifier(prot_expr, reps=2)
result_prot = clf_prot.run_all(pvals=True, slope_pvals=True, n_permutations=200, n_jobs=1)

print("Classifying RNA-seq layer …")
clf_rna    = Classifier(rna_expr, reps=2)
result_rna = clf_rna.run_all(pvals=True, slope_pvals=True, n_permutations=200, n_jobs=1)

print("\nProteomics labels:")
print(result_prot["label"].value_counts().to_string())
print("\nRNA-seq labels:")
print(result_rna["label"].value_counts().to_string(), "\n")

# ---------------------------------------------------------------------------
# 3. Per-layer classification overview
# ---------------------------------------------------------------------------
print("Section 1: Per-layer label distributions …")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
viz.label_distribution(result_prot, ax=axes[0], title=LAYER_LABELS[0])
viz.label_distribution(result_rna,  ax=axes[1], title=LAYER_LABELS[1])
xmax = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
for ax in axes:
    ax.set_xlim(0, xmax)
plt.tight_layout()
out = FIGURES / "31_layer_label_distributions.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 4. Cross-layer statistical comparison
#
# compare_conditions joins on shared feature IDs.  Effect-size columns
# (delta_tau, delta_pirs, delta_phase) and significance tests
# (tau_padj, phase_padj) are computed for all 200 shared features.
#
# Column naming convention: _A = proteomics, _B = RNA-seq.
#   rhythmicity_status "lost"   = rhythmic in proteomics, non-rhythmic in RNA
#   rhythmicity_status "gained" = non-rhythmic in proteomics, rhythmic in RNA
# ---------------------------------------------------------------------------
print("Section 2: Cross-layer statistical comparison …")

comparison = compare_conditions(result_prot, result_rna)

print("\nRhythmicity status (proteomics → RNA-seq):")
print(comparison["rhythmicity_status"].value_counts().to_string())
print("\nLabel transition table (proteomics → RNA-seq):")
print(label_change_table(comparison).to_string())

sig_tau   = comparison["tau_padj"].lt(0.05).sum()
sig_phase = comparison["phase_padj"].lt(0.05).sum() if "phase_padj" in comparison.columns else 0
print(f"\nSignificant τ changes   (tau_padj  < 0.05): {sig_tau}")
print(f"Significant phase shifts (phase_padj < 0.05): {sig_phase}\n")

fig = viz.comparison_summary(
    comparison,
    outpath=str(FIGURES / "32_cross_layer_summary.png"),
)
plt.close(fig)
print(f"  saved → {FIGURES / '32_cross_layer_summary.png'}")

# ---------------------------------------------------------------------------
# 5. Divergent features: prot-only vs. RNA-only rhythmic
# ---------------------------------------------------------------------------
print("Section 3: Divergent features …")

prot_only = comparison[comparison["rhythmicity_status"] == "lost"]
rna_only  = comparison[comparison["rhythmicity_status"] == "gained"]

print(f"\n  Rhythmic at protein level only ({len(prot_only)} features):")
if not prot_only.empty:
    print(prot_only[["tau_mean_A", "tau_mean_B", "delta_tau", "tau_padj"]].head(5).to_string())

print(f"\n  Rhythmic at mRNA level only ({len(rna_only)} features):")
if not rna_only.empty:
    print(rna_only[["tau_mean_A", "tau_mean_B", "delta_tau", "tau_padj"]].head(5).to_string())

fig, ax = plt.subplots(figsize=(5, 5))
viz.rhythmicity_shift_scatter(comparison, ax=ax)
ax.set_xlabel(f"TauMean — {LAYER_LABELS[0]}")
ax.set_ylabel(f"TauMean — {LAYER_LABELS[1]}")
ax.set_title("Rhythmicity across molecular layers")
plt.tight_layout()
out = FIGURES / "33_cross_layer_rhythmicity_scatter.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved → {out}")

# ---------------------------------------------------------------------------
# 6. Peptide-level workflow: aggregate_to_protein
#
# If your proteomics classification result carries a (Peptide, Protein)
# MultiIndex — for example from running PIRS directly on peptide-level data
# before protein aggregation — call aggregate_to_protein() to collapse it
# to one row per protein before comparing.
#
# Aggregation rules applied by aggregate_to_protein:
#   - numeric columns : mean across peptides
#   - phase_mean      : circular mean (handles wrap-around correctly)
#   - label           : majority vote across peptides
#
# This section constructs a synthetic peptide-level result by expanding the
# protein-level result and adding per-peptide noise.
# ---------------------------------------------------------------------------
print("\nSection 4: aggregate_to_protein workflow …")

rng = np.random.default_rng(42)
n_pep_per = rng.integers(2, 4, size=len(result_prot))

peptide_result = result_prot.loc[result_prot.index.repeat(n_pep_per)].copy()
for col in ("tau_mean", "pirs_score"):
    noise = rng.normal(0, 0.02, size=len(peptide_result))
    peptide_result[col] = (peptide_result[col].values + noise).clip(min=0)

prot_ids = peptide_result.index
pep_ids  = [f"pep_{i:04d}" for i in range(len(peptide_result))]
peptide_result.index = pd.MultiIndex.from_arrays(
    [pep_ids, prot_ids], names=["Peptide", "Protein"]
)
n_prot = peptide_result.index.get_level_values("Protein").nunique()
print(f"  Peptide-level result: {len(peptide_result)} peptides → {n_prot} proteins")

# Passing peptide-level data to compare_conditions raises a descriptive error:
try:
    compare_conditions(peptide_result, result_rna)
except ValueError as e:
    print(f"  Raised ValueError as expected:\n    {e}")

# Aggregate to protein level, then compare:
prot_agg = aggregate_to_protein(peptide_result)
comparison_from_pep = compare_conditions(prot_agg, result_rna)
print(f"  aggregate_to_protein + compare_conditions: "
      f"{len(comparison_from_pep)} features compared successfully")

if "--show" in sys.argv:
    plt.show()

print(f"\nDone. Figures written to {FIGURES}")
