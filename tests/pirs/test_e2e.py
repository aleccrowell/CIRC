"""End-to-end tests for the PIRS pipeline.

Exercises the full ranker pipeline — scoring, PIRS permutation p-values, and
slope permutation p-values — on a synthetic dataset with a known mix of
constitutive and linearly trending genes.  Assertions are at the distributional
level (group means, column presence) rather than exact values.
"""
import numpy as np
import pandas as pd
import pytest

from circ.pirs.rank import ranker


@pytest.fixture(scope="module")
def e2e_file(tmp_path_factory):
    """10 constitutive + 5 linear genes, 6 timepoints, 3 reps."""
    rng = np.random.default_rng(77)
    tpoints = [2, 4, 6, 8, 10, 12]
    n_reps = 3
    cols = [f"CT{t:02d}_{r}" for t in tpoints for r in range(1, n_reps + 1)]
    data = {}
    for i in range(10):
        data[f"const_{i}"] = rng.normal(5.0 + i * 0.3, 0.1, len(cols))
    for i in range(5):
        data[f"linear_{i}"] = np.array(
            [float(t * (2 + i * 0.4)) + rng.normal(0, 0.2) for t in tpoints for _ in range(n_reps)]
        )
    df = pd.DataFrame(data, index=cols).T
    df.index.name = "#"
    path = tmp_path_factory.mktemp("pirs_e2e") / "e2e.txt"
    df.to_csv(path, sep="\t")
    return str(path)


@pytest.fixture(scope="module")
def e2e_ranker(e2e_file):
    """Ranker with scores, PIRS p-values, and slope p-values pre-computed."""
    r = ranker(e2e_file, anova=False)
    r.get_tpoints()
    r.calculate_scores()
    r.calculate_pvals(n_permutations=199)
    r.calculate_slope_pvals(n_permutations=199)
    return r


class TestPirsE2EScoring:
    def test_all_genes_scored(self, e2e_ranker):
        assert len(e2e_ranker.errors) == 15

    def test_scores_sorted_ascending(self, e2e_ranker):
        assert e2e_ranker.errors["score"].is_monotonic_increasing

    def test_constitutive_genes_score_lower_on_average(self, e2e_ranker):
        errors = e2e_ranker.errors
        const_scores = errors.loc[[g for g in errors.index if "const" in g], "score"]
        linear_scores = errors.loc[[g for g in errors.index if "linear" in g], "score"]
        assert const_scores.mean() < linear_scores.mean()

    def test_top_ranked_are_constitutive(self, e2e_ranker):
        top = e2e_ranker.errors.index[:8]
        assert all("const" in g for g in top)


class TestPirsE2EPvals:
    def test_pval_columns_present(self, e2e_ranker):
        for col in ("pval", "pval_bh"):
            assert col in e2e_ranker.errors.columns

    def test_pvals_in_unit_interval(self, e2e_ranker):
        for col in ("pval", "pval_bh"):
            vals = e2e_ranker.errors[col]
            assert (vals >= 0).all() and (vals <= 1).all()

    def test_linear_genes_have_lower_pval_on_average(self, e2e_ranker):
        errors = e2e_ranker.errors
        const_pvals = errors.loc[[g for g in errors.index if "const" in g], "pval"]
        linear_pvals = errors.loc[[g for g in errors.index if "linear" in g], "pval"]
        assert linear_pvals.mean() < const_pvals.mean()


class TestPirsE2ESlopePvals:
    def test_slope_pval_columns_present(self, e2e_ranker):
        for col in ("slope_pval", "slope_pval_bh"):
            assert col in e2e_ranker.errors.columns

    def test_slope_pvals_in_unit_interval(self, e2e_ranker):
        for col in ("slope_pval", "slope_pval_bh"):
            vals = e2e_ranker.errors[col]
            assert (vals >= 0).all() and (vals <= 1).all()

    def test_linear_genes_have_lower_slope_pval_on_average(self, e2e_ranker):
        errors = e2e_ranker.errors
        const_pvals = errors.loc[[g for g in errors.index if "const" in g], "slope_pval"]
        linear_pvals = errors.loc[[g for g in errors.index if "linear" in g], "slope_pval"]
        assert linear_pvals.mean() < const_pvals.mean()


class TestPirsE2EFileOutput:
    def test_pirs_sort_writes_all_columns(self, e2e_file, tmp_path):
        out = str(tmp_path / "scores.txt")
        r = ranker(e2e_file, anova=False)
        r.pirs_sort(outname=out, pvals=True, slope_pvals=True, n_permutations=99)
        written = pd.read_csv(out, sep="\t", index_col=0)
        for col in ("score", "pval", "pval_bh", "slope_pval", "slope_pval_bh"):
            assert col in written.columns

    def test_output_row_count_matches_input(self, e2e_file, tmp_path):
        out = str(tmp_path / "scores.txt")
        r = ranker(e2e_file, anova=False)
        r.pirs_sort(outname=out, pvals=True, slope_pvals=True, n_permutations=99)
        written = pd.read_csv(out, sep="\t", index_col=0)
        assert len(written) == 15
