"""Full proteomics pipeline: impute → batch-correct → classify → visualize.

Use case:
    You have a label-free proteomics dataset with missing values and known
    (or suspected) batch effects from TMT pooling.  This example walks through
    the complete circ.limbr pipeline — KNN imputation, SVA batch correction,
    and expression classification — then validates the results against
    simulation ground truth.

Run:
    poetry run python examples/04_proteomics_pipeline.py

Intermediate files (imputed, normalized, pool map) are written to a temp
directory and cleaned up automatically.  Figures are saved to ./figures/.

Key differences from the simple expression examples:
  - tpoints=12 required — SVA's circular-correlation window needs ≥ 12 points
  - Two index columns (Peptide, Protein) instead of one (#)
  - imputable handles NULL missing values and deduplicates peptides → proteins
  - sva pool-normalizes pooled controls, then removes latent batch effects
  - Classifier receives the protein-level normalized output

Swap the simulation block for real data:
    from circ.io import read_expression
    # Provide a tab-separated file with Peptide, Protein, ZT* columns and NULL
    # for missing values, plus a pool_map.parquet matching samples to controls.
    imp = imputable("your_raw_data.txt", missingness=0.3, neighbors=10)
    imp.impute_data("imputed.parquet")
    sva_obj = sva("imputed.parquet", design="c", data_type="p",
                  pool="pool_map.parquet")
    ...
"""

import sys, os, time, tempfile
from pathlib import Path

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from circ.simulations import simulate
from circ.limbr.imputation import imputable
from circ.limbr.batch_fx import sva
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
# 1. Simulate a proteomics-style dataset with batch effects and missing values
# ---------------------------------------------------------------------------
# tpoints=12 is required: SVA's circ_cor() shifts the per-timepoint means
# matrix by 12 and 6 positions, so fewer than 12 unique timepoints causes all
# correlations to alias to zero and the SVA reduction step produces empty data.
print("Simulating proteomics data …")
t0 = time.time()
sim = simulate(
    tpoints=12, nrows=200, nreps=2, tpoint_space=2,
    pcirc=0.25, plin=0.15,
    n_batch_effects=2, pbatch=0.5, effect_size=2.0,
    p_miss=0.25, lam_miss=4,
    rseed=42,
)
print(f"  {sim.nrows} proteins, 12 × 2 = 24 samples, "
      f"2 batch effects, 25 % missing  ({time.time()-t0:.1f}s)")

with tempfile.TemporaryDirectory(prefix="circ_prot_") as tmp:

    pool_stem = os.path.join(tmp, "pool_map")
    sim_stem  = os.path.join(tmp, "sim")

    sim.generate_pool_map(out_name=pool_stem)
    sim.write_proteomics(out_name=sim_stem)

    noise_file     = sim_stem + "_with_noise.txt"
    pool_file      = pool_stem + ".parquet"
    true_class_file = sim_stem + "_true_classes.txt"
    imputed_path   = os.path.join(tmp, "imputed.parquet")
    norm_path      = os.path.join(tmp, "normalized.parquet")

    # -----------------------------------------------------------------------
    # 2. Impute missing values with KNN
    # -----------------------------------------------------------------------
    print("Imputing missing values …")
    t0 = time.time()
    imp = imputable(noise_file, missingness=0.3, neighbors=5)
    imp.impute_data(imputed_path)
    imputed_df = pd.read_parquet(imputed_path)
    print(f"  {imputed_df.shape[0]} peptide groups retained "
          f"({time.time()-t0:.1f}s)")

    # -----------------------------------------------------------------------
    # 3. SVA batch-effect removal
    # -----------------------------------------------------------------------
    print("Running SVA batch correction …")
    t0 = time.time()
    sva_obj = sva(imputed_path, design="c", data_type="p", pool=pool_file)
    sva_obj.preprocess_default()
    sva_obj.perm_test(nperm=100, npr=1)
    sva_obj.output_default(norm_path)
    n_sv = int((sva_obj.sigs < 0.05).sum()) if hasattr(sva_obj, "sigs") else "?"
    print(f"  {n_sv} significant surrogate variable(s) removed "
          f"({time.time()-t0:.1f}s)")

    # -----------------------------------------------------------------------
    # 4. Classify normalized expression
    # -----------------------------------------------------------------------
    print("Classifying …")
    t0 = time.time()
    clf = Classifier(norm_path, reps=2)
    result = clf.run_all(
        pvals=True,
        slope_pvals=True,
        n_permutations=200,
        n_jobs=1,
    )
    print(f"  ({time.time()-t0:.1f}s)")
    print(result["label"].value_counts().to_string(), "\n")

    # Load ground-truth labels (index values are the same Protein IDs as
    # the normalized data, but the index name differs — rename to align)
    true_classes = (
        pd.read_csv(true_class_file, sep="\t", index_col=0)
        .rename_axis("#")
    )

    # -----------------------------------------------------------------------
    # 5. Before / after batch-correction QC
    #
    # Per-sample medians reveal systematic column-level offsets introduced by
    # batch effects (e.g., all samples from batch 2 are shifted up).  After
    # SVA normalization the medians should be much more uniform across samples.
    # -----------------------------------------------------------------------
    print("Section 1: Before/after normalization QC …")

    # Aggregate imputed peptides → protein-level medians for the "before" view
    zt_cols = [c for c in sva_obj.svd_norm.columns
               if str(c).startswith(("ZT", "CT"))]

    before_protein = imputed_df.groupby(level="Protein").median()
    before_medians = before_protein[zt_cols].median(axis=0)
    after_medians  = sva_obj.svd_norm[zt_cols].median(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, medians, title in [
        (axes[0], before_medians, "After imputation — pre-SVA"),
        (axes[1], after_medians,  "After SVA normalization"),
    ]:
        x = range(len(medians))
        ax.bar(x, medians.values, color="#4878CF", alpha=0.7, width=0.8)
        ax.axhline(medians.mean(), color="#D65F5F", ls="--", lw=0.9,
                   label=f"mean={medians.mean():.2f}")
        ax.set_xticks(list(x))
        ax.set_xticklabels(medians.index.tolist(), rotation=90, fontsize=6)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Median protein expression")
        ax.set_title(title)
        cv = medians.std() / medians.mean() * 100
        ax.text(0.98, 0.97, f"CV={cv:.1f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=8)
        ax.legend(frameon=False, fontsize=8)
        sns.despine(ax=ax)

    fig.suptitle("Batch correction QC: per-sample median expression", y=1.01)
    plt.tight_layout()
    out = FIGURES / "15_normalization_qc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")

    # -----------------------------------------------------------------------
    # 6. Classification results
    # -----------------------------------------------------------------------
    print("Section 2: Classification results …")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    viz.label_distribution(result, ax=axes[0])
    viz.pirs_vs_tau(result, ax=axes[1])
    viz.top_constitutive_candidates(result, n_top=15, ax=axes[2])
    plt.tight_layout()
    out = FIGURES / "16_classification_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")

    # Mean expression profiles using the normalized data
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    viz.mean_expression_profiles(
        sva_obj.svd_norm, result, ax=axes[0],
        title="Mean profile by label (normalized)",
    )
    viz.phase_amplitude_scatter(result, ax=axes[1])
    plt.tight_layout()
    out = FIGURES / "17_profiles_and_phase.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")

    # -----------------------------------------------------------------------
    # 7. Benchmark against simulation ground truth
    # -----------------------------------------------------------------------
    print("Section 3: Benchmark …")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    viz.classification_pr(
        result, true_classes,
        ground_truth_col="Const",
        score_col="pirs_score",
        invert_score=True,
        ax=axes[0],
        title="Constitutive detection PR",
    )
    viz.classification_roc(
        result, true_classes, ax=axes[1],
        title="ROC: all classification tasks",
    )
    plt.tight_layout()
    out = FIGURES / "18_benchmark.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")

    # Adaptive summary figure for the full pipeline result
    print("Adaptive summary …")
    fig = viz.classification_summary(
        result,
        outpath=str(FIGURES / "19_proteomics_summary.png"),
    )
    plt.close(fig)
    print(f"  saved → {FIGURES / '29_proteomics_summary.png'}")

    # Interactive HTML
    if _HAS_PLOTLY:
        print("Interactive …")
        fig_html = iviz.top_constitutive_candidates(result, n_top=30)
        out = FIGURES / "20_interactive_proteomics_candidates.html"
        fig_html.write_html(str(out))
        print(f"  saved → {out}")

if "--show" in sys.argv:
    plt.show()

print("\nDone. Figures written to", FIGURES)
