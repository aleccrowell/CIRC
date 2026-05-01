"""Tests for circ.visualization.benchmarks — plotting functions only.

Metric correctness tests (roc_auc, classification_auc, classification_ap)
live in tests/test_evaluation.py.
"""

import re

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.axes

from circ.visualization.benchmarks import (
    pr_curve,
    roc_curve_plot,
    classification_pr,
    classification_roc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def curves_df():
    """Synthetic PR curves DataFrame."""
    rng = np.random.default_rng(0)
    rows = []
    for method in ["PIRS", "RSD"]:
        for rep in range(3):
            n = 20
            recall = np.linspace(0, 1, n)
            precision = np.clip(rng.uniform(0.3, 1.0, n) - recall * 0.3, 0, 1)
            for r, p in zip(recall, precision):
                rows.append({"recall": r, "precision": p, "method": method, "rep": rep})
    return pd.DataFrame(rows)


@pytest.fixture
def merged_dict():
    """Synthetic merged dict for roc_curve_plot tests."""
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
# pr_curve
# ---------------------------------------------------------------------------


class TestPrCurve:
    def test_returns_axes(self, curves_df):
        ax = pr_curve(curves_df)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_accepts_explicit_ax(self, curves_df):
        _, ax = plt.subplots()
        returned = pr_curve(curves_df, ax=ax)
        assert returned is ax

    def test_baseline_drawn(self, curves_df):
        ax = pr_curve(curves_df, baseline=0.4)
        _, labels = ax.get_legend_handles_labels()
        assert "random classifier" in labels

    def test_saves_to_file(self, curves_df, tmp_path):
        import os

        out = str(tmp_path / "pr.png")
        pr_curve(curves_df, outpath=out)
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# roc_curve_plot
# ---------------------------------------------------------------------------


class TestRocCurvePlot:
    def test_returns_axes(self, merged_dict):
        ax = roc_curve_plot(merged_dict)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_one_line_per_method(self, merged_dict):
        ax = roc_curve_plot(merged_dict)
        n_methods = len(merged_dict)
        # each method + diagonal = n_methods + 1 lines
        assert len(ax.lines) == n_methods + 1

    def test_accepts_truth_score_columns(self):
        rng = np.random.default_rng(4)
        n = 50
        d = {
            "m1": pd.DataFrame(
                {"truth": rng.integers(0, 2, n), "score": rng.uniform(0, 1, n)}
            )
        }
        ax = roc_curve_plot(d)
        assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# classification_pr
# ---------------------------------------------------------------------------


class TestClassificationPr:
    def test_returns_axes(self, classification_df, true_classes):
        ax = classification_pr(classification_df, true_classes)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_no_crash_on_unmatched_index(self, classification_df, true_classes):
        subset = true_classes.iloc[:10].copy()
        ax = classification_pr(classification_df, subset)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_circadian_truth_col(self, classification_df, true_classes):
        ax = classification_pr(
            classification_df,
            true_classes,
            ground_truth_col="Circadian",
            score_col="tau_mean",
            invert_score=False,
        )
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_empty_intersection_no_crash(self, classification_df, true_classes):
        tc = true_classes.copy()
        tc.index = [f"other_{i}" for i in range(len(tc))]
        ax = classification_pr(classification_df, tc)
        assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# classification_roc — rendering
# ---------------------------------------------------------------------------


class TestClassificationRoc:
    def test_returns_axes(self, classification_df, true_classes):
        ax = classification_roc(classification_df, true_classes)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_auto_task_detection(self, classification_df, true_classes):
        ax = classification_roc(classification_df, true_classes)
        # pirs_score→Const, tau_mean→Circadian, emp_p→Circadian,
        # pval→Const, slope_pval→Linear, + diagonal = at least 6 lines
        assert len(ax.lines) >= 3

    def test_explicit_tasks(self, classification_df, true_classes):
        tasks = [("pirs_score", "Const", True)]
        ax = classification_roc(classification_df, true_classes, tasks=tasks)
        # 1 method line + diagonal
        assert len(ax.lines) == 2


# ---------------------------------------------------------------------------
# classification_roc — AUC values embedded in legend labels
# ---------------------------------------------------------------------------


class TestClassificationRocValues:
    """AUC values written into classification_roc() legend labels should be correct."""

    def _extract_auc(self, ax):
        """Return the first AUC value found in any line label on the axes."""
        for line in ax.lines:
            m = re.search(r"AUC=([0-9.]+)", line.get_label())
            if m:
                return float(m.group(1))
        return None

    def test_perfect_slope_pval_gives_auc_near_one(self):
        """A score that perfectly separates Linear genes should appear as AUC ≈ 1.0."""
        n = 100
        clf_df = pd.DataFrame(
            {"slope_pval": np.array([0.001] * 25 + [0.99] * 75)},
            index=[f"g{i}" for i in range(n)],
        )
        truth_df = pd.DataFrame(
            {"Linear": np.array([1] * 25 + [0] * 75)},
            index=clf_df.index,
        )
        ax = classification_roc(
            clf_df, truth_df, tasks=[("slope_pval", "Linear", True)]
        )
        auc_val = self._extract_auc(ax)
        assert auc_val is not None, (
            "AUC value not found in classification_roc legend labels"
        )
        assert auc_val > 0.99, (
            f"Perfect slope_pval should give AUC > 0.99; got {auc_val:.3f}"
        )

    def test_random_score_gives_near_chance_auc(self):
        """A random score should appear as AUC ≈ 0.5 in the legend."""
        rng = np.random.default_rng(55)
        n = 300
        clf_df = pd.DataFrame(
            {"pirs_score": rng.uniform(0, 1, n)},
            index=[f"g{i}" for i in range(n)],
        )
        truth_df = pd.DataFrame(
            {"Const": rng.integers(0, 2, n)},
            index=clf_df.index,
        )
        ax = classification_roc(clf_df, truth_df, tasks=[("pirs_score", "Const", True)])
        auc_val = self._extract_auc(ax)
        assert auc_val is not None, (
            "AUC value not found in classification_roc legend labels"
        )
        assert abs(auc_val - 0.5) < 0.1, (
            f"Random pirs_score AUC={auc_val:.3f} should be near 0.5"
        )

    def test_all_tasks_produce_auc_labels(self, classification_df, true_classes):
        """All auto-detected tasks should embed an AUC value in their legend label."""
        ax = classification_roc(classification_df, true_classes)
        auc_labels = [
            l
            for line in ax.lines
            for l in [line.get_label()]
            if re.search(r"AUC=([0-9.]+)", l)
        ]
        # Should have at least one labelled task line (random diagonal also has AUC=0.5)
        assert len(auc_labels) >= 2, (
            f"Expected ≥2 lines with AUC labels; found {len(auc_labels)}"
        )
