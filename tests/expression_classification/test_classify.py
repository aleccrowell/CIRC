"""Tests for circ.expression_classification.classify."""
import os
import types

import numpy as np
import pandas as pd
import pytest

from circ.expression_classification.classify import Classifier, _make_pipeline_args, _REF_DIR


# ---------------------------------------------------------------------------
# Unit tests: PIRS integration
# ---------------------------------------------------------------------------

class TestRunPirs:
    def test_returns_dataframe(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file)
        scores = clf.run_pirs()
        assert isinstance(scores, pd.DataFrame)

    def test_score_column_present(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file)
        scores = clf.run_pirs()
        assert 'score' in scores.columns

    def test_all_genes_scored_without_anova(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file, anova=False)
        scores = clf.run_pirs()
        raw = pd.read_csv(simulated_expression_file, sep='\t', index_col=0)
        assert len(scores) == len(raw)

    def test_scores_are_non_negative(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file)
        scores = clf.run_pirs()
        assert (scores['score'] >= 0).all()

    def test_stores_pirs_scores(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file)
        clf.run_pirs()
        assert clf.pirs_scores is not None

    def test_pvals_adds_columns(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file)
        scores = clf.run_pirs(pvals=True, n_permutations=49)
        assert 'pval' in scores.columns
        assert 'pval_bh' in scores.columns

    def test_slope_pvals_adds_columns(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file)
        scores = clf.run_pirs(slope_pvals=True, n_permutations=49)
        assert 'slope_pval' in scores.columns
        assert 'slope_pval_bh' in scores.columns

    def test_pvals_in_unit_interval(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file)
        scores = clf.run_pirs(pvals=True, slope_pvals=True, n_permutations=49)
        for col in ('pval', 'pval_bh', 'slope_pval', 'slope_pval_bh'):
            assert (scores[col] >= 0).all() and (scores[col] <= 1).all()

    def test_no_pval_columns_without_flags(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file)
        scores = clf.run_pirs()
        assert 'pval' not in scores.columns
        assert 'slope_pval' not in scores.columns


# ---------------------------------------------------------------------------
# Unit tests: BooteJTK integration
# ---------------------------------------------------------------------------

class TestRunBootjtk:
    def test_returns_dataframe(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file, size=10, reps=2)
        results = clf.run_bootjtk()
        assert isinstance(results, pd.DataFrame)

    def test_tau_mean_column_present(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file, size=10, reps=2)
        results = clf.run_bootjtk()
        assert 'TauMean' in results.columns

    def test_gamma_bh_column_present(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file, size=10, reps=2)
        results = clf.run_bootjtk()
        assert 'GammaBH' in results.columns

    def test_result_covers_all_genes(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file, size=10, reps=2)
        results = clf.run_bootjtk()
        raw = pd.read_csv(simulated_expression_file, sep='\t', index_col=0)
        assert len(results) == len(raw)

    def test_no_temp_files_left_behind(self, simulated_expression_file, tmp_path):
        before = set(os.listdir(tmp_path))
        clf = Classifier(simulated_expression_file, size=10, reps=2)
        clf.run_bootjtk()
        after = set(os.listdir(tmp_path))
        assert before == after

    def test_stores_rhythm_results(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file, size=10, reps=2)
        clf.run_bootjtk()
        assert clf.rhythm_results is not None


# ---------------------------------------------------------------------------
# Unit tests: classify()
# ---------------------------------------------------------------------------

class TestClassify:
    @pytest.fixture(autouse=True)
    def _prep(self, simulated_expression_file):
        self.clf = Classifier(simulated_expression_file, size=10, reps=2)
        self.clf.run_pirs()
        self.clf.run_bootjtk()

    def test_returns_dataframe(self):
        result = self.clf.classify()
        assert isinstance(result, pd.DataFrame)

    def test_label_column_present(self):
        result = self.clf.classify()
        assert 'label' in result.columns

    def test_valid_labels_only(self):
        result = self.clf.classify()
        valid = {'constitutive', 'rhythmic', 'variable', 'noisy_rhythmic', 'unclassified'}
        assert set(result['label'].unique()).issubset(valid)

    def test_all_four_columns_present(self):
        result = self.clf.classify()
        for col in ('pirs_score', 'tau_mean', 'emp_p', 'label'):
            assert col in result.columns

    def test_pirs_score_matches_run_pirs(self):
        result = self.clf.classify()
        shared = self.clf.pirs_scores.index.intersection(result.index)
        pd.testing.assert_series_equal(
            result.loc[shared, 'pirs_score'],
            self.clf.pirs_scores.loc[shared, 'score'].rename('pirs_score'),
            check_names=False,
        )

    def test_tau_mean_matches_bootjtk(self):
        result = self.clf.classify()
        shared = self.clf.rhythm_results.index.intersection(result.index)
        pd.testing.assert_series_equal(
            result.loc[shared, 'tau_mean'],
            self.clf.rhythm_results.loc[shared, 'TauMean'].rename('tau_mean'),
            check_names=False,
        )

    def test_raises_without_pirs(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file)
        clf.run_bootjtk = lambda: None  # skip BooteJTK
        clf.rhythm_results = pd.DataFrame({'TauMean': [0.5]}, index=['g1'])
        with pytest.raises(RuntimeError, match='run_pirs'):
            clf.classify()

    def test_raises_without_bootjtk(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file)
        clf.pirs_scores = pd.DataFrame({'score': [0.1]}, index=['g1'])
        with pytest.raises(RuntimeError, match='run_bootjtk'):
            clf.classify()

    def test_pirs_percentile_affects_stable_fraction(self):
        result_25 = self.clf.classify(pirs_percentile=25)
        result_75 = self.clf.classify(pirs_percentile=75)
        stable_labels = {'constitutive', 'rhythmic'}
        n_stable_25 = result_25['label'].isin(stable_labels).sum()
        n_stable_75 = result_75['label'].isin(stable_labels).sum()
        assert n_stable_25 <= n_stable_75

    def test_tau_threshold_affects_rhythmic_fraction(self):
        result_low = self.clf.classify(tau_threshold=0.1, emp_p_threshold=1.0)
        result_high = self.clf.classify(tau_threshold=0.9, emp_p_threshold=1.0)
        rhythmic_labels = {'rhythmic', 'noisy_rhythmic'}
        n_low = result_low['label'].isin(rhythmic_labels).sum()
        n_high = result_high['label'].isin(rhythmic_labels).sum()
        assert n_low >= n_high

    def test_slope_pval_columns_forwarded_to_classify(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file, size=10, reps=2)
        clf.run_pirs(slope_pvals=True, n_permutations=49)
        clf.run_bootjtk()
        result = clf.classify()
        assert 'slope_pval' in result.columns
        assert 'slope_pval_bh' in result.columns

    def test_linear_label_emitted_with_slope_pvals(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file, size=10, reps=2)
        clf.run_pirs(slope_pvals=True, n_permutations=49)
        clf.run_bootjtk()
        result = clf.classify(slope_pval_threshold=1.0)  # flag everything as sloped
        valid = {'constitutive', 'rhythmic', 'linear', 'variable', 'noisy_rhythmic'}
        assert set(result['label'].unique()).issubset(valid)
        assert 'linear' in result['label'].values

    def test_linear_label_absent_without_slope_pvals(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file, size=10, reps=2)
        clf.run_pirs()
        clf.run_bootjtk()
        result = clf.classify()
        assert 'linear' not in result['label'].values

    def test_slope_threshold_affects_linear_count(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file, size=10, reps=2)
        clf.run_pirs(slope_pvals=True, n_permutations=49)
        clf.run_bootjtk()
        result_strict = clf.classify(slope_pval_threshold=0.001)
        result_lenient = clf.classify(slope_pval_threshold=1.0)
        n_strict = (result_strict['label'] == 'linear').sum()
        n_lenient = (result_lenient['label'] == 'linear').sum()
        assert n_strict <= n_lenient


# ---------------------------------------------------------------------------
# Integration test: run_all()
# ---------------------------------------------------------------------------

class TestRunAll:
    def test_run_all_returns_dataframe(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file, size=10, reps=2)
        result = clf.run_all()
        assert isinstance(result, pd.DataFrame)

    def test_run_all_sets_all_attributes(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file, size=10, reps=2)
        clf.run_all()
        assert clf.pirs_scores is not None
        assert clf.rhythm_results is not None
        assert clf.classifications is not None

    def test_run_all_label_column(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file, size=10, reps=2)
        result = clf.run_all()
        assert 'label' in result.columns

    def test_run_all_no_unlabelled_rows(self, simulated_expression_file):
        clf = Classifier(simulated_expression_file, size=10, reps=2)
        result = clf.run_all()
        assert result['label'].notna().all()


# ---------------------------------------------------------------------------
# Unit tests: _make_pipeline_args helper
# ---------------------------------------------------------------------------

class TestMakePipelineArgs:
    def test_returns_namespace(self):
        args = _make_pipeline_args(
            filename='/tmp/data.txt',
            ref_dir=_REF_DIR,
            reps=2,
            size=50,
            workers=1,
            basic=True,
        )
        assert isinstance(args, types.SimpleNamespace)

    def test_period_file_exists(self):
        args = _make_pipeline_args(
            filename='/tmp/data.txt',
            ref_dir=_REF_DIR,
            reps=2,
            size=50,
            workers=1,
            basic=True,
        )
        assert os.path.isfile(args.period), f"Missing ref file: {args.period}"

    def test_phase_file_exists(self):
        args = _make_pipeline_args(
            filename='/tmp/data.txt',
            ref_dir=_REF_DIR,
            reps=2,
            size=50,
            workers=1,
            basic=True,
        )
        assert os.path.isfile(args.phase), f"Missing ref file: {args.phase}"

    def test_width_file_exists(self):
        args = _make_pipeline_args(
            filename='/tmp/data.txt',
            ref_dir=_REF_DIR,
            reps=2,
            size=50,
            workers=1,
            basic=True,
        )
        assert os.path.isfile(args.width), f"Missing ref file: {args.width}"

    def test_reps_and_size_set(self):
        args = _make_pipeline_args(
            filename='/tmp/data.txt',
            ref_dir=_REF_DIR,
            reps=3,
            size=20,
            workers=2,
            basic=False,
        )
        assert args.reps == 3
        assert args.size == 20
        assert args.workers == 2
        assert args.basic is False
