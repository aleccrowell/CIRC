"""Tests for circ.evaluation — pure metric functions."""

import numpy as np
import pandas as pd
import pytest

from circ.evaluation import roc_auc, classification_auc, classification_ap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def merged_dict():
    """Synthetic merged dict for roc_auc tests."""
    rng = np.random.default_rng(1)
    n = 100
    truth = rng.integers(0, 2, n)
    return {
        "method_a": pd.DataFrame({"Circadian": truth, "GammaBH": rng.uniform(0, 1, n)}),
        "method_b": pd.DataFrame({"Circadian": truth, "GammaBH": rng.uniform(0, 1, n)}),
    }


@pytest.fixture
def classification_df():
    """Synthetic classification output."""
    rng = np.random.default_rng(2)
    n = 80
    labels = (
        ["constitutive"] * 20
        + ["rhythmic"] * 20
        + ["variable"] * 20
        + ["noisy_rhythmic"] * 20
    )
    return pd.DataFrame(
        {
            "pirs_score": np.abs(rng.normal(0, 1, n)),
            "tau_mean": rng.uniform(0, 1, n),
            "emp_p": rng.uniform(0, 1, n),
            "pval": rng.uniform(0, 1, n),
            "slope_pval": rng.uniform(0, 1, n),
            "label": labels,
        },
        index=[f"g{i}" for i in range(n)],
    )


@pytest.fixture
def true_classes(classification_df):
    """Simulated ground-truth binary labels aligned with classification_df."""
    rng = np.random.default_rng(3)
    n = len(classification_df)
    return pd.DataFrame(
        {
            "Const": rng.integers(0, 2, n),
            "Circadian": rng.integers(0, 2, n),
            "Linear": rng.integers(0, 2, n),
        },
        index=classification_df.index,
    )


# ---------------------------------------------------------------------------
# roc_auc — structural
# ---------------------------------------------------------------------------


class TestRocAuc:
    def test_returns_dict(self, merged_dict):
        result = roc_auc(merged_dict)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(merged_dict.keys())

    def test_auc_in_unit_interval(self, merged_dict):
        result = roc_auc(merged_dict)
        for v in result.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# roc_auc — metric correctness
# ---------------------------------------------------------------------------


class TestRocAucQuality:
    """roc_auc() should return correct values for controllable inputs.

    roc_auc() computes 1 - GammaBH before scoring, so a low GammaBH for
    positive-class genes yields a high discriminating score (as expected for
    a p-value).
    """

    def _make_dict(self, truth, gammabh, tag="m"):
        return {tag: pd.DataFrame({"Circadian": truth, "GammaBH": gammabh})}

    def test_perfect_score_gives_auc_one(self):
        """GammaBH == 0 for all positive genes → AUC should be 1.0."""
        truth = np.array([1] * 30 + [0] * 30)
        gammabh = np.array([0.0] * 30 + [1.0] * 30)
        result = roc_auc(self._make_dict(truth, gammabh))
        assert result["m"] == pytest.approx(1.0)

    def test_inverted_score_gives_auc_zero(self):
        """GammaBH == 1 for all positive genes → AUC should be 0.0."""
        truth = np.array([1] * 30 + [0] * 30)
        gammabh = np.array([1.0] * 30 + [0.0] * 30)
        result = roc_auc(self._make_dict(truth, gammabh))
        assert result["m"] == pytest.approx(0.0)

    def test_random_score_gives_auc_near_half(self):
        """A random score should give AUC close to 0.5."""
        rng = np.random.default_rng(99)
        n = 400
        truth = rng.integers(0, 2, n)
        gammabh = rng.uniform(0, 1, n)
        result = roc_auc(self._make_dict(truth, gammabh))
        assert abs(result["m"] - 0.5) < 0.1, (
            f"Random-score AUC={result['m']:.3f} should be near 0.5"
        )

    def test_multiple_methods_ranked_correctly(self):
        """Perfect score should outrank a random score."""
        n = 100
        truth = np.array([1] * 50 + [0] * 50)
        rng = np.random.default_rng(7)
        d = {
            "perfect": pd.DataFrame(
                {"Circadian": truth, "GammaBH": np.array([0.0] * 50 + [1.0] * 50)}
            ),
            "random": pd.DataFrame(
                {"Circadian": truth, "GammaBH": rng.uniform(0, 1, n)}
            ),
        }
        result = roc_auc(d)
        assert result["perfect"] > result["random"], (
            f"perfect AUC={result['perfect']:.3f} should exceed "
            f"random AUC={result['random']:.3f}"
        )


# ---------------------------------------------------------------------------
# classification_auc
# ---------------------------------------------------------------------------


class TestClassificationAuc:
    def test_returns_dict(self, classification_df, true_classes):
        result = classification_auc(classification_df, true_classes)
        assert isinstance(result, dict)

    def test_keys_are_score_truth_tuples(self, classification_df, true_classes):
        result = classification_auc(classification_df, true_classes)
        for key in result:
            assert isinstance(key, tuple) and len(key) == 2

    def test_values_in_unit_interval(self, classification_df, true_classes):
        result = classification_auc(classification_df, true_classes)
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_auto_detection_finds_expected_tasks(self, classification_df, true_classes):
        result = classification_auc(classification_df, true_classes)
        assert ("pirs_score", "Const") in result
        assert ("tau_mean", "Circadian") in result
        assert ("slope_pval", "Linear") in result

    def test_explicit_tasks_respected(self, classification_df, true_classes):
        result = classification_auc(
            classification_df,
            true_classes,
            tasks=[("slope_pval", "Linear", True)],
        )
        assert list(result.keys()) == [("slope_pval", "Linear")]

    def test_perfect_score_gives_auc_one(self):
        """A score that perfectly separates classes should give AUC=1.0."""
        n = 60
        clf_df = pd.DataFrame(
            {"pirs_score": np.array([0.0] * 30 + [1.0] * 30)},
            index=[f"g{i}" for i in range(n)],
        )
        truth_df = pd.DataFrame(
            {"Const": np.array([1] * 30 + [0] * 30)},
            index=clf_df.index,
        )
        result = classification_auc(
            clf_df, truth_df, tasks=[("pirs_score", "Const", True)]
        )
        assert result[("pirs_score", "Const")] == pytest.approx(1.0)

    def test_empty_intersection_omitted(self):
        """Tasks with no overlapping index should be omitted from the result."""
        clf_df = pd.DataFrame({"pirs_score": [0.1, 0.2]}, index=["a", "b"])
        truth_df = pd.DataFrame({"Const": [1, 0]}, index=["x", "y"])
        result = classification_auc(
            clf_df, truth_df, tasks=[("pirs_score", "Const", True)]
        )
        assert result == {}


# ---------------------------------------------------------------------------
# classification_ap
# ---------------------------------------------------------------------------


class TestClassificationAp:
    def test_returns_float(self, classification_df, true_classes):
        ap = classification_ap(classification_df, true_classes)
        assert isinstance(ap, float)

    def test_value_in_unit_interval(self, classification_df, true_classes):
        ap = classification_ap(classification_df, true_classes)
        assert 0.0 <= ap <= 1.0

    def test_perfect_score_gives_ap_one(self):
        """A score that perfectly separates classes should give AP=1.0."""
        n = 60
        clf_df = pd.DataFrame(
            {"pirs_score": np.array([0.0] * 30 + [1.0] * 30)},
            index=[f"g{i}" for i in range(n)],
        )
        truth_df = pd.DataFrame(
            {"Const": np.array([1] * 30 + [0] * 30)},
            index=clf_df.index,
        )
        ap = classification_ap(
            clf_df,
            truth_df,
            ground_truth_col="Const",
            score_col="pirs_score",
            invert_score=True,
        )
        assert ap == pytest.approx(1.0)

    def test_empty_intersection_returns_none(self):
        clf_df = pd.DataFrame({"pirs_score": [0.1, 0.2]}, index=["a", "b"])
        truth_df = pd.DataFrame({"Const": [1, 0]}, index=["x", "y"])
        ap = classification_ap(clf_df, truth_df)
        assert ap is None

    def test_non_default_score_col(self, classification_df, true_classes):
        ap = classification_ap(
            classification_df,
            true_classes,
            ground_truth_col="Circadian",
            score_col="tau_mean",
            invert_score=False,
        )
        assert ap is not None
        assert 0.0 <= ap <= 1.0
