"""End-to-end pipeline test: simulation → imputation → SVA → classification → visualization.

Mirrors the README 'End-to-end example' using a small synthetic proteomics
dataset with batch effects and missing values.  The full pipeline runs once via
a module-scoped fixture; individual test classes check each stage.

Assertions are structural (column presence, shape, label validity) and
directional (known-class averages) rather than exact numeric values.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from circ.simulations import simulate
from circ.limbr.imputation import imputable
from circ.limbr.batch_fx import sva
from circ.expression_classification.classify import Classifier
import circ.visualization as viz


# ---------------------------------------------------------------------------
# Module-scoped fixture: run the full pipeline once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run the complete pipeline and return all intermediate artifacts."""
    tmp = tmp_path_factory.mktemp("e2e_pipeline")

    # 1. Simulate
    # tpoints=12 is required: SVA's circ_cor() shifts by 12 and 6 in the
    # per-timepoint means matrix, so fewer than 12 unique timepoints causes
    # all correlations to alias to zero and data_reduced becomes empty.
    sim = simulate(
        tpoints=12, nrows=100, nreps=2, tpoint_space=2,
        pcirc=0.3, plin=0.2,
        n_batch_effects=1, pbatch=0.5, effect_size=2.0,
        p_miss=0.2, lam_miss=3,
        rseed=42,
    )
    pool_map_stem = str(tmp / "pool_map")
    sim_stem = str(tmp / "sim")
    sim.generate_pool_map(out_name=pool_map_stem)
    sim.write_proteomics(out_name=sim_stem)

    # 2. Impute
    imputed_path = str(tmp / "imputed.parquet")
    imp = imputable(sim_stem + "_with_noise.txt", missingness=0.3, neighbors=5)
    imp.impute_data(imputed_path)

    # 3. SVA batch correction
    normalized_path = str(tmp / "normalized.parquet")
    sva_obj = sva(
        imputed_path,
        design="c",
        data_type="p",
        pool=pool_map_stem + ".parquet",
    )
    sva_obj.preprocess_default()
    sva_obj.perm_test(nperm=99)
    sva_obj.output_default(normalized_path)

    # 4. Classify
    clf = Classifier(normalized_path, reps=2, size=10)
    result = clf.run_all(slope_pvals=True, n_permutations=99)

    true_classes = pd.read_csv(sim_stem + "_true_classes.txt", sep="\t", index_col=0)

    return {
        "tmp": tmp,
        "sim": sim,
        "sim_stem": sim_stem,
        "pool_map_stem": pool_map_stem,
        "imputed_path": imputed_path,
        "normalized_path": normalized_path,
        "sva_obj": sva_obj,
        "clf": clf,
        "result": result,
        "true_classes": true_classes,
    }


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Step 1: Simulation
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_classes_array_length(self, pipeline):
        assert len(pipeline["sim"].classes) == 100

    def test_class_labels_valid(self, pipeline):
        assert set(pipeline["sim"].classes).issubset({"circadian", "linear", "constitutive"})

    def test_all_three_classes_present(self, pipeline):
        assert set(pipeline["sim"].classes) == {"circadian", "linear", "constitutive"}

    def test_sim_miss_has_nan(self, pipeline):
        assert np.isnan(pipeline["sim"].sim_miss).any()

    def test_with_noise_file_exists(self, pipeline):
        assert os.path.exists(pipeline["sim_stem"] + "_with_noise.txt")

    def test_true_classes_file_exists(self, pipeline):
        assert os.path.exists(pipeline["sim_stem"] + "_true_classes.txt")

    def test_pool_map_file_exists(self, pipeline):
        assert os.path.exists(pipeline["pool_map_stem"] + ".parquet")

    def test_true_classes_columns(self, pipeline):
        assert {"Circadian", "Linear", "Const"}.issubset(pipeline["true_classes"].columns)

    def test_true_classes_binary(self, pipeline):
        for col in ("Circadian", "Linear", "Const"):
            assert set(pipeline["true_classes"][col].unique()).issubset({0, 1})

    def test_class_fractions_roughly_correct(self, pipeline):
        classes = pipeline["sim"].classes
        circ_frac = (classes == "circadian").mean()
        lin_frac = (classes == "linear").mean()
        assert 0.1 < circ_frac < 0.6
        assert 0.0 < lin_frac < 0.5


# ---------------------------------------------------------------------------
# Step 2: Imputation
# ---------------------------------------------------------------------------

class TestImputation:
    def test_output_file_exists(self, pipeline):
        assert os.path.exists(pipeline["imputed_path"])

    def test_no_missing_after_imputation(self, pipeline):
        df = pd.read_parquet(pipeline["imputed_path"])
        zt_cols = [c for c in df.columns if str(c).startswith("ZT")]
        assert not df[zt_cols].isnull().any().any()

    def test_row_count_at_most_input(self, pipeline):
        df = pd.read_parquet(pipeline["imputed_path"])
        assert len(df) <= 100

    def test_zt_columns_present(self, pipeline):
        df = pd.read_parquet(pipeline["imputed_path"])
        zt_cols = [c for c in df.columns if str(c).startswith("ZT")]
        assert len(zt_cols) == 24  # 12 tpoints * 2 reps


# ---------------------------------------------------------------------------
# Step 3: Normalization (SVA)
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_output_file_exists(self, pipeline):
        assert os.path.exists(pipeline["normalized_path"])

    def test_svd_norm_attribute_set(self, pipeline):
        assert pipeline["sva_obj"].svd_norm is not None

    def test_svd_norm_is_dataframe(self, pipeline):
        assert isinstance(pipeline["sva_obj"].svd_norm, pd.DataFrame)

    def test_normalized_has_no_missing(self, pipeline):
        assert not pipeline["sva_obj"].svd_norm.isnull().any().any()

    def test_diagnostic_sidecars_exist(self, pipeline):
        stem = pipeline["normalized_path"].replace(".parquet", "")
        for suffix in ("_trends", "_perms", "_tks"):
            path = stem + suffix + ".parquet"
            assert os.path.exists(path), f"missing sidecar: {path}"


# ---------------------------------------------------------------------------
# Step 4: Classification
# ---------------------------------------------------------------------------

class TestClassification:
    def test_result_is_dataframe(self, pipeline):
        assert isinstance(pipeline["result"], pd.DataFrame)

    def test_result_is_non_empty(self, pipeline):
        assert len(pipeline["result"]) > 0

    def test_required_columns_present(self, pipeline):
        for col in ("pirs_score", "tau_mean", "label"):
            assert col in pipeline["result"].columns, f"missing column: {col}"

    def test_slope_pval_columns_present(self, pipeline):
        for col in ("slope_pval", "slope_pval_bh"):
            assert col in pipeline["result"].columns

    def test_all_rows_labelled(self, pipeline):
        assert pipeline["result"]["label"].notna().all()

    def test_valid_label_set(self, pipeline):
        valid = {"constitutive", "rhythmic", "linear", "variable", "noisy_rhythmic", "unclassified"}
        assert set(pipeline["result"]["label"].unique()).issubset(valid)

    def test_pirs_scores_non_negative(self, pipeline):
        assert (pipeline["result"]["pirs_score"] >= 0).all()

    def test_slope_pvals_in_unit_interval(self, pipeline):
        for col in ("slope_pval", "slope_pval_bh"):
            vals = pipeline["result"][col]
            assert (vals >= 0).all() and (vals <= 1).all()

    def test_classifier_attributes_populated(self, pipeline):
        clf = pipeline["clf"]
        assert clf.pirs_scores is not None
        assert clf.rhythm_results is not None
        assert clf.classifications is not None

    def test_constitutive_label_has_lower_pirs_than_variable(self, pipeline):
        """Classifier consistency: 'constitutive' genes must have lower PIRS than 'variable'.

        This is guaranteed by the PIRS percentile cutoff in classify(), so it
        tests that the classifier logic applied correctly end-to-end.
        """
        result = pipeline["result"]
        const_ids = result[result["label"] == "constitutive"].index
        variable_ids = result[result["label"] == "variable"].index
        if len(const_ids) < 3 or len(variable_ids) < 3:
            pytest.skip("insufficient label representation for comparison")
        assert (result.loc[const_ids, "pirs_score"].mean() <
                result.loc[variable_ids, "pirs_score"].mean())

    def test_linear_genes_have_lower_slope_pval_on_average(self, pipeline):
        """Genes simulated as linear should have lower slope_pval on average."""
        result = pipeline["result"]
        true_classes = pipeline["true_classes"]
        shared = result.index.intersection(true_classes.index)
        if len(shared) < 10:
            pytest.skip("too few matched genes for directional test")
        tc = true_classes.loc[shared]
        linear_ids = tc[tc["Linear"] == 1].index
        const_ids = tc[tc["Const"] == 1].index
        if len(linear_ids) < 3 or len(const_ids) < 3:
            pytest.skip("insufficient class representation")
        assert (result.loc[linear_ids, "slope_pval"].mean() <
                result.loc[const_ids, "slope_pval"].mean())


# ---------------------------------------------------------------------------
# Step 5: Visualization — classification plots
# ---------------------------------------------------------------------------

class TestClassificationPlots:
    def test_label_distribution(self, pipeline):
        assert isinstance(viz.label_distribution(pipeline["result"]), plt.Axes)

    def test_pirs_vs_tau(self, pipeline):
        assert isinstance(viz.pirs_vs_tau(pipeline["result"]), plt.Axes)

    def test_volcano(self, pipeline):
        assert isinstance(viz.volcano(pipeline["result"]), plt.Axes)

    def test_pirs_score_distribution(self, pipeline):
        assert isinstance(viz.pirs_score_distribution(pipeline["result"]), plt.Axes)

    def test_tau_pval_scatter(self, pipeline):
        assert isinstance(viz.tau_pval_scatter(pipeline["result"]), plt.Axes)

    def test_slope_pval_scatter(self, pipeline):
        assert isinstance(viz.slope_pval_scatter(pipeline["result"]), plt.Axes)

    def test_slope_vs_rhythm(self, pipeline):
        assert isinstance(viz.slope_vs_rhythm(pipeline["result"]), plt.Axes)

    def test_phase_wheel(self, pipeline):
        assert isinstance(viz.phase_wheel(pipeline["result"]), plt.Axes)

    def test_period_distribution(self, pipeline):
        assert isinstance(viz.period_distribution(pipeline["result"]), plt.Axes)

    def test_classification_summary_returns_figure(self, pipeline):
        fig = viz.classification_summary(pipeline["result"])
        assert isinstance(fig, plt.Figure)

    def test_classification_summary_saves_file(self, pipeline):
        outpath = str(pipeline["tmp"] / "summary.png")
        viz.classification_summary(pipeline["result"], outpath=outpath)
        assert os.path.exists(outpath)

    def test_custom_panel_composition(self, pipeline):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        viz.label_distribution(pipeline["result"], ax=axes[0])
        viz.pirs_vs_tau(pipeline["result"], ax=axes[1])
        viz.phase_wheel(pipeline["result"], ax=axes[2])
        assert len(fig.axes) == 3


# ---------------------------------------------------------------------------
# Step 6: Visualization — benchmark plots against ground truth
# ---------------------------------------------------------------------------

class TestBenchmarkPlots:
    def test_classification_pr_returns_axes(self, pipeline):
        ax = viz.classification_pr(
            pipeline["result"],
            pipeline["true_classes"],
            ground_truth_col="Const",
        )
        assert isinstance(ax, plt.Axes)

    def test_classification_roc_returns_axes(self, pipeline):
        ax = viz.classification_roc(
            pipeline["result"],
            pipeline["true_classes"],
        )
        assert isinstance(ax, plt.Axes)

    def test_classification_roc_explicit_tasks(self, pipeline):
        ax = viz.classification_roc(
            pipeline["result"],
            pipeline["true_classes"],
            tasks=[
                ("tau_mean", "Circadian", False),
                ("slope_pval", "Linear", True),
            ],
        )
        assert isinstance(ax, plt.Axes)

    def test_benchmark_figure_saves(self, pipeline):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        viz.classification_pr(
            pipeline["result"], pipeline["true_classes"],
            ground_truth_col="Const", ax=axes[0],
        )
        viz.classification_roc(
            pipeline["result"], pipeline["true_classes"],
            ax=axes[1],
        )
        outpath = str(pipeline["tmp"] / "benchmark.png")
        fig.savefig(outpath, dpi=72, bbox_inches="tight")
        assert os.path.exists(outpath)
