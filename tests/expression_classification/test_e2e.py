"""End-to-end tests for the expression classification pipeline.

Exercises Classifier.run_all() with permutation p-values on simulated data
with known ground-truth class labels (constitutive / linear / circadian).
Verifies that the five-label scheme is populated correctly and that the
slope p-values are directionally consistent with the known gene classes.
"""
import numpy as np
import pandas as pd
import pytest

from circ.expression_classification.classify import Classifier


@pytest.fixture(scope="module")
def e2e_data(tmp_path_factory):
    """Simulated dataset with known class labels.

    Returns (expression_file, true_classes_df) where true_classes_df has
    columns Circadian / Linear / Const (1/0 indicators).
    """
    from circ.simulations import simulate

    tmp = tmp_path_factory.mktemp("classify_e2e")
    out = str(tmp / "e2e_expr.txt")
    sim = simulate(
        tpoints=6, nrows=60, nreps=2, tpoint_space=4,
        pcirc=0.3, plin=0.3, rseed=11,
    )
    sim.write_output(out_name=out)
    true_classes = pd.read_csv(out[:-4] + "_true_classes.txt", sep="\t", index_col=0)
    return out, true_classes


class TestClassifierE2EBasic:
    """Full pipeline without permutation p-values (fast path)."""

    def test_run_all_returns_dataframe(self, e2e_data):
        expr_file, _ = e2e_data
        clf = Classifier(expr_file, size=10, reps=2)
        result = clf.run_all()
        assert isinstance(result, pd.DataFrame)

    def test_all_rows_labelled(self, e2e_data):
        expr_file, _ = e2e_data
        clf = Classifier(expr_file, size=10, reps=2)
        result = clf.run_all()
        assert result["label"].notna().all()

    def test_labels_from_four_class_scheme(self, e2e_data):
        expr_file, _ = e2e_data
        clf = Classifier(expr_file, size=10, reps=2)
        result = clf.run_all()
        valid = {"constitutive", "rhythmic", "variable", "noisy_rhythmic", "unclassified"}
        assert set(result["label"].unique()).issubset(valid)
        assert "linear" not in result["label"].values

    def test_pirs_score_column_present(self, e2e_data):
        expr_file, _ = e2e_data
        clf = Classifier(expr_file, size=10, reps=2)
        result = clf.run_all()
        assert "pirs_score" in result.columns

    def test_all_attributes_populated(self, e2e_data):
        expr_file, _ = e2e_data
        clf = Classifier(expr_file, size=10, reps=2)
        clf.run_all()
        assert clf.pirs_scores is not None
        assert clf.rhythm_results is not None
        assert clf.classifications is not None


class TestClassifierE2EWithSlopePvals:
    """Full pipeline with slope permutation p-values — five-label scheme."""

    @pytest.fixture(scope="class")
    def result_with_slope(self, e2e_data):
        expr_file, _ = e2e_data
        clf = Classifier(expr_file, size=10, reps=2)
        result = clf.run_all(slope_pvals=True, n_permutations=199)
        return result

    def test_returns_dataframe(self, result_with_slope):
        assert isinstance(result_with_slope, pd.DataFrame)

    def test_all_rows_labelled(self, result_with_slope):
        assert result_with_slope["label"].notna().all()

    def test_slope_pval_columns_present(self, result_with_slope):
        assert "slope_pval" in result_with_slope.columns
        assert "slope_pval_bh" in result_with_slope.columns

    def test_slope_pvals_in_unit_interval(self, result_with_slope):
        for col in ("slope_pval", "slope_pval_bh"):
            vals = result_with_slope[col]
            assert (vals >= 0).all() and (vals <= 1).all()

    def test_five_class_scheme_used(self, result_with_slope):
        valid = {"constitutive", "rhythmic", "linear", "variable", "noisy_rhythmic", "unclassified"}
        assert set(result_with_slope["label"].unique()).issubset(valid)

    def test_linear_label_appears_with_lenient_threshold(self, e2e_data):
        """Forcing slope_pval_threshold=1.0 classifies all non-rhythmic genes as linear."""
        expr_file, _ = e2e_data
        clf = Classifier(expr_file, size=10, reps=2)
        result = clf.run_all(slope_pvals=True, n_permutations=99, slope_pval_threshold=1.0)
        assert "linear" in result["label"].values

    def test_known_linear_genes_have_lower_slope_pval(self, e2e_data, result_with_slope):
        _, true_classes = e2e_data
        shared = result_with_slope.index.intersection(true_classes.index)
        linear_ids = true_classes.loc[shared][true_classes.loc[shared, "Linear"] == 1].index
        const_ids = true_classes.loc[shared][true_classes.loc[shared, "Const"] == 1].index
        if len(linear_ids) >= 3 and len(const_ids) >= 3:
            assert (result_with_slope.loc[linear_ids, "slope_pval"].mean() <
                    result_with_slope.loc[const_ids, "slope_pval"].mean())


class TestClassifierE2EWithAllPvals:
    """run_all with both pvals and slope_pvals."""

    def test_all_pval_columns_present(self, e2e_data):
        expr_file, _ = e2e_data
        clf = Classifier(expr_file, size=10, reps=2)
        result = clf.run_all(pvals=True, slope_pvals=True, n_permutations=99)
        for col in ("pval", "pval_bh", "slope_pval", "slope_pval_bh"):
            assert col in result.columns

    def test_all_pvals_in_unit_interval(self, e2e_data):
        expr_file, _ = e2e_data
        clf = Classifier(expr_file, size=10, reps=2)
        result = clf.run_all(pvals=True, slope_pvals=True, n_permutations=99)
        for col in ("pval", "pval_bh", "slope_pval", "slope_pval_bh"):
            assert (result[col] >= 0).all() and (result[col] <= 1).all()
