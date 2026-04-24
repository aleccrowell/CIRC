import pytest
import numpy as np
import pandas as pd
import os

from circ.pirs.rank import ranker, rsd_ranker
from tests.pirs.conftest import make_ranker_tsv


class TestRankerInit:
    def test_loads_data(self, tsv_file):
        r = ranker(tsv_file)
        assert isinstance(r.data, pd.DataFrame)
        assert len(r.data) > 0

    def test_drops_all_zero_rows(self, tmp_path):
        df = pd.DataFrame(
            {"CT02_1": [1.0, 0.0], "CT04_1": [2.0, 0.0]},
            index=["gene_a", "all_zeros"],
        )
        df.index.name = "#"
        path = tmp_path / "zeros.txt"
        df.to_csv(path, sep="\t")
        r = ranker(str(path))
        assert "all_zeros" not in r.data.index

    def test_anova_flag_default_true(self, tsv_file):
        r = ranker(tsv_file)
        assert r.anova is True

    def test_anova_flag_false(self, tsv_file):
        r = ranker(tsv_file, anova=False)
        assert r.anova is False


class TestGetTpoints:
    def test_ct_prefix_columns(self, tsv_file):
        r = ranker(tsv_file)
        r.get_tpoints()
        assert hasattr(r, "tpoints")
        assert isinstance(r.tpoints, np.ndarray)
        assert len(r.tpoints) == len(r.data.columns)

    def test_no_ct_prefix_columns(self, tsv_no_ct):
        r = ranker(tsv_no_ct)
        r.get_tpoints()
        assert len(r.tpoints) == len(r.data.columns)

    def test_tpoints_values(self, tsv_file):
        r = ranker(tsv_file)
        r.get_tpoints()
        unique_tpoints = sorted(np.unique(r.tpoints).tolist())
        assert unique_tpoints == [2, 4, 6, 8]

    def test_tpoints_replicated(self, tsv_file):
        r = ranker(tsv_file)
        r.get_tpoints()
        for t in np.unique(r.tpoints):
            assert np.sum(r.tpoints == t) == 3


class TestRemoveAnova:
    def test_removes_rhythmic_gene(self, tsv_file):
        r = ranker(tsv_file)
        r.get_tpoints()
        before = len(r.data)
        r.remove_anova(alpha=0.05)
        assert len(r.data) < before
        assert "rhythmic_gene" not in r.data.index

    def test_keeps_constitutive_genes(self, tsv_file):
        r = ranker(tsv_file)
        r.get_tpoints()
        r.remove_anova(alpha=0.05)
        for i in range(8):
            assert f"gene_{i}" in r.data.index

    def test_alpha_controls_stringency(self, tsv_file):
        r_strict = ranker(tsv_file)
        r_strict.get_tpoints()
        r_strict.remove_anova(alpha=1e-10)

        r_lenient = ranker(tsv_file)
        r_lenient.get_tpoints()
        r_lenient.remove_anova(alpha=0.99)

        assert len(r_strict.data) >= len(r_lenient.data)


class TestCalculateScores:
    def test_returns_dataframe(self, tsv_file):
        r = ranker(tsv_file)
        r.get_tpoints()
        scores = r.calculate_scores()
        assert isinstance(scores, pd.DataFrame)

    def test_score_column_present(self, tsv_file):
        r = ranker(tsv_file)
        r.get_tpoints()
        scores = r.calculate_scores()
        assert "score" in scores.columns

    def test_index_matches_data(self, tsv_file):
        r = ranker(tsv_file)
        r.get_tpoints()
        scores = r.calculate_scores()
        assert set(scores.index) == set(r.data.index)

    def test_scores_sorted_ascending(self, tsv_file):
        r = ranker(tsv_file)
        r.get_tpoints()
        scores = r.calculate_scores()
        assert scores["score"].is_monotonic_increasing

    def test_constitutive_scores_lower_than_rhythmic(self, tmp_path):
        n_reps = 3
        tpoints = [2, 4, 6, 8, 10, 12]
        cols = [f"CT{t:02d}_{r}" for t in tpoints for r in range(1, n_reps + 1)]
        flat = np.ones(len(cols))
        rhythmic = np.array(
            [float(t_idx * 10) for t_idx in range(len(tpoints)) for _ in range(n_reps)]
        )
        df = pd.DataFrame({"flat": flat, "rhythmic": rhythmic}, index=cols).T
        df.index.name = "#"
        path = tmp_path / "simple.txt"
        df.to_csv(path, sep="\t")

        r = ranker(str(path), anova=False)
        r.get_tpoints()
        scores = r.calculate_scores()
        assert scores.loc["flat", "score"] < scores.loc["rhythmic", "score"]


class TestPirsSort:
    def test_returns_dataframe(self, tsv_file):
        r = ranker(tsv_file)
        result = r.pirs_sort()
        assert isinstance(result, pd.DataFrame)

    def test_output_is_subset_of_input(self, tsv_file):
        r = ranker(tsv_file)
        original_index = set(pd.read_csv(tsv_file, sep="\t", index_col=0).index)
        result = r.pirs_sort()
        assert set(result.index).issubset(original_index)

    def test_columns_preserved(self, tsv_file):
        r = ranker(tsv_file)
        result = r.pirs_sort()
        original = pd.read_csv(tsv_file, sep="\t", index_col=0)
        assert list(result.columns) == list(original.columns)

    def test_anova_false_keeps_rhythmic(self, tsv_file):
        r = ranker(tsv_file, anova=False)
        result = r.pirs_sort()
        assert "rhythmic_gene" in result.index

    def test_anova_true_removes_rhythmic(self, tsv_file):
        r = ranker(tsv_file, anova=True)
        result = r.pirs_sort()
        assert "rhythmic_gene" not in result.index

    def test_outname_writes_scores_file(self, tsv_file, tmp_path):
        out = str(tmp_path / "scores.txt")
        r = ranker(tsv_file)
        r.pirs_sort(outname=out)
        assert os.path.exists(out)
        written = pd.read_csv(out, sep="\t", index_col=0)
        assert "score" in written.columns


class TestRsdRanker:
    def test_init_loads_data(self, tsv_file):
        r = rsd_ranker(tsv_file)
        assert isinstance(r.data, pd.DataFrame)
        assert len(r.data) > 0

    def test_drops_all_zero_rows(self, tmp_path):
        df = pd.DataFrame(
            {"CT02_1": [1.0, 0.0], "CT04_1": [2.0, 0.0]},
            index=["gene_a", "all_zeros"],
        )
        df.index.name = "#"
        path = tmp_path / "zeros.txt"
        df.to_csv(path, sep="\t")
        r = rsd_ranker(str(path))
        assert "all_zeros" not in r.data.index

    def test_calculate_scores_returns_dataframe(self, tsv_file):
        r = rsd_ranker(tsv_file)
        scores = r.calculate_scores()
        assert isinstance(scores, pd.DataFrame)
        assert "score" in scores.columns

    def test_scores_sorted_ascending(self, tsv_file):
        r = rsd_ranker(tsv_file)
        scores = r.calculate_scores()
        assert scores["score"].is_monotonic_increasing

    def test_index_matches_data(self, tsv_file):
        r = rsd_ranker(tsv_file)
        scores = r.calculate_scores()
        assert set(scores.index) == set(r.data.index)

    def test_constitutive_lower_rsd_than_variable(self, tmp_path):
        cols = ["CT02_1", "CT04_1", "CT06_1", "CT08_1"]
        df = pd.DataFrame(
            {
                "const": [1.0, 1.0, 1.0, 1.0],
                "variable": [0.1, 5.0, 0.2, 8.0],
            },
            index=cols,
        ).T
        df.index.name = "#"
        path = tmp_path / "simple.txt"
        df.to_csv(path, sep="\t")
        r = rsd_ranker(str(path))
        scores = r.calculate_scores()
        assert scores.loc["const", "score"] < scores.loc["variable", "score"]

    def test_rsd_sort_returns_dataframe(self, tsv_file):
        r = rsd_ranker(tsv_file)
        result = r.rsd_sort()
        assert isinstance(result, pd.DataFrame)

    def test_rsd_sort_columns_preserved(self, tsv_file):
        r = rsd_ranker(tsv_file)
        result = r.rsd_sort()
        original = pd.read_csv(tsv_file, sep="\t", index_col=0)
        assert list(result.columns) == list(original.columns)

    def test_rsd_sort_outname_writes_file(self, tsv_file, tmp_path):
        out = str(tmp_path / "rsd_scores.txt")
        r = rsd_ranker(tsv_file)
        r.rsd_sort(outname=out)
        assert os.path.exists(out)
        written = pd.read_csv(out, sep="\t", index_col=0)
        assert "score" in written.columns
