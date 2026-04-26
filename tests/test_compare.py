"""Tests for circ.compare — condition comparison statistics."""
import numpy as np
import pandas as pd
import pytest

from circ.compare import compare_conditions, label_change_table, _circular_diff, _bh_correct


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(rseed=0, with_uncertainty=False, n=100, index_prefix='gene'):
    rng = np.random.default_rng(rseed)
    idx = pd.Index([f'{index_prefix}_{i:04d}' for i in range(n)], name='#')
    tau   = rng.uniform(0.0, 1.0, n)
    emp_p = np.where(tau > 0.5, rng.uniform(0.0, 0.05, n), rng.uniform(0.05, 1.0, n))
    pirs  = rng.uniform(0.0, 2.0, n)
    phase = rng.uniform(0.0, 24.0, n)
    labels = np.where(
        (tau > 0.5) & (emp_p < 0.05), 'rhythmic',
        np.where(pirs < np.percentile(pirs, 50), 'constitutive', 'variable'),
    )
    df = pd.DataFrame({
        'tau_mean':   tau,
        'emp_p':      emp_p,
        'pirs_score': pirs,
        'phase_mean': phase,
        'label':      labels,
    }, index=idx)
    if with_uncertainty:
        df['tau_std']   = rng.uniform(0.05, 0.2, n)
        df['phase_std'] = rng.uniform(0.5, 2.0, n)
        df['n_boots']   = 50
    return df


# ---------------------------------------------------------------------------
# _circular_diff
# ---------------------------------------------------------------------------

class TestCircularDiff:
    def test_zero_diff(self):
        assert _circular_diff(6.0, 6.0) == pytest.approx(0.0)

    def test_positive_diff(self):
        assert _circular_diff(2.0, 6.0) == pytest.approx(4.0)

    def test_negative_diff(self):
        assert _circular_diff(6.0, 2.0) == pytest.approx(-4.0)

    def test_wrap_forward(self):
        # 22 → 2 should give +4 h (not −20 h)
        assert _circular_diff(22.0, 2.0) == pytest.approx(4.0)

    def test_wrap_backward(self):
        # 2 → 22 should give −4 h (not +20 h)
        assert _circular_diff(2.0, 22.0) == pytest.approx(-4.0)

    def test_exactly_half_period(self):
        # ±12 h boundary: difference of exactly 12 wraps to +12
        diff = _circular_diff(0.0, 12.0)
        assert abs(diff) == pytest.approx(12.0)

    def test_vectorised(self):
        a = np.array([0.0, 22.0, 12.0])
        b = np.array([6.0,  2.0, 18.0])
        result = _circular_diff(a, b)
        assert result == pytest.approx([6.0, 4.0, 6.0])


# ---------------------------------------------------------------------------
# _bh_correct
# ---------------------------------------------------------------------------

class TestBHCorrect:
    def test_all_nan(self):
        result = _bh_correct([np.nan, np.nan])
        assert np.all(np.isnan(result))

    def test_single_value(self):
        result = _bh_correct([0.03])
        assert result[0] == pytest.approx(0.03)

    def test_bounded_zero_to_one(self):
        pvals = [0.001, 0.01, 0.05, 0.1, 0.5]
        result = _bh_correct(pvals)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_monotone_in_rank(self):
        # After BH correction the adjusted p-values should be non-decreasing
        # when sorted by original p-value.
        pvals = np.array([0.001, 0.01, 0.02, 0.04, 0.05, 0.2])
        result = _bh_correct(pvals)
        assert np.all(np.diff(result) >= -1e-12)

    def test_nan_passthrough(self):
        result = _bh_correct([0.01, np.nan, 0.05])
        assert np.isnan(result[1])
        assert not np.isnan(result[0])
        assert not np.isnan(result[2])


# ---------------------------------------------------------------------------
# compare_conditions
# ---------------------------------------------------------------------------

class TestCompareConditions:
    def test_returns_only_shared_genes(self):
        A = _make_result(rseed=0, n=80)
        # B has 80 genes but only 60 overlap with A
        B = _make_result(rseed=1, n=80)
        result = compare_conditions(A, B)
        shared = A.index.intersection(B.index)
        assert list(result.index) == list(shared)

    def test_partial_overlap(self):
        A = _make_result(rseed=0, n=100, index_prefix='gene')
        B_idx = pd.Index([f'gene_{i:04d}' for i in range(50, 150)], name='#')
        B = _make_result(rseed=1, n=100)
        B.index = B_idx
        result = compare_conditions(A, B)
        assert len(result) == 50

    def test_required_columns(self):
        A = _make_result(rseed=0)
        B = _make_result(rseed=1)
        result = compare_conditions(A, B)
        for col in (
            'label_A', 'label_B', 'rhythmicity_status',
            'tau_mean_A', 'tau_mean_B', 'delta_tau',
            'pirs_score_A', 'pirs_score_B', 'delta_pirs',
            'phase_A', 'phase_B', 'delta_phase',
            'emp_p_A', 'emp_p_B',
        ):
            assert col in result.columns, f"Missing column: {col}"

    def test_delta_tau_correct(self):
        A = _make_result(rseed=0)
        B = _make_result(rseed=1)
        result = compare_conditions(A, B)
        shared = A.index.intersection(B.index)
        expected = B.loc[shared, 'tau_mean'].values - A.loc[shared, 'tau_mean'].values
        np.testing.assert_allclose(result['delta_tau'].values, expected)

    def test_delta_pirs_correct(self):
        A = _make_result(rseed=0)
        B = _make_result(rseed=1)
        result = compare_conditions(A, B)
        shared = A.index.intersection(B.index)
        expected = B.loc[shared, 'pirs_score'].values - A.loc[shared, 'pirs_score'].values
        np.testing.assert_allclose(result['delta_pirs'].values, expected)

    def test_delta_phase_wrapped(self):
        A = _make_result(rseed=0)
        B = _make_result(rseed=1)
        result = compare_conditions(A, B)
        valid = result['delta_phase'].dropna()
        assert (valid >= -12).all() and (valid <= 12).all()

    def test_delta_phase_nan_for_nonrhythmic(self):
        A = _make_result(rseed=0)
        B = _make_result(rseed=1)
        result = compare_conditions(A, B)
        non_both = result['rhythmicity_status'] != 'maintained_rhythmic'
        assert result.loc[non_both, 'delta_phase'].isna().all()

    def test_no_shared_genes_raises(self):
        A = _make_result(rseed=0, index_prefix='gene')
        B = _make_result(rseed=1, index_prefix='prot')
        with pytest.raises(ValueError, match='share no gene IDs'):
            compare_conditions(A, B)

    def test_rhythmicity_status_values(self):
        A = _make_result(rseed=0)
        B = _make_result(rseed=1)
        result = compare_conditions(A, B)
        valid = {'gained', 'lost', 'maintained_rhythmic', 'maintained_nonrhythmic'}
        assert set(result['rhythmicity_status'].unique()).issubset(valid)

    def test_gained_lost_consistency(self):
        """Genes rhythmic only in B → gained; rhythmic only in A → lost."""
        A = _make_result(rseed=0)
        B = _make_result(rseed=1)
        result = compare_conditions(A, B)
        gained = result[result['rhythmicity_status'] == 'gained']
        lost   = result[result['rhythmicity_status'] == 'lost']
        # Gained genes should have low emp_p in B and higher in A
        if not gained.empty and 'emp_p_A' in gained.columns:
            assert (gained['emp_p_B'] < 0.05).all()
        if not lost.empty and 'emp_p_A' in lost.columns:
            assert (lost['emp_p_A'] < 0.05).all()

    def test_significance_columns_with_uncertainty(self):
        A = _make_result(rseed=0, with_uncertainty=True)
        B = _make_result(rseed=1, with_uncertainty=True)
        result = compare_conditions(A, B)
        for col in ('tau_pval', 'tau_padj', 'phase_pval', 'phase_padj'):
            assert col in result.columns, f"Missing significance column: {col}"

    def test_no_significance_without_uncertainty(self):
        A = _make_result(rseed=0)
        B = _make_result(rseed=1)
        result = compare_conditions(A, B)
        for col in ('tau_pval', 'tau_padj'):
            assert col not in result.columns

    def test_tau_padj_bounded(self):
        A = _make_result(rseed=0, with_uncertainty=True)
        B = _make_result(rseed=1, with_uncertainty=True)
        result = compare_conditions(A, B)
        padj = result['tau_padj'].dropna()
        assert (padj >= 0).all() and (padj <= 1).all()

    def test_phase_pval_only_for_rhythmic_both(self):
        A = _make_result(rseed=0, with_uncertainty=True)
        B = _make_result(rseed=1, with_uncertainty=True)
        result = compare_conditions(A, B)
        if 'phase_pval' in result.columns:
            non_both = result['rhythmicity_status'] != 'maintained_rhythmic'
            assert result.loc[non_both, 'phase_pval'].isna().all()

    def test_no_phase_columns_without_phase_data(self):
        A = _make_result(rseed=0).drop(columns=['phase_mean'])
        B = _make_result(rseed=1).drop(columns=['phase_mean'])
        result = compare_conditions(A, B)
        assert 'delta_phase' not in result.columns


# ---------------------------------------------------------------------------
# label_change_table
# ---------------------------------------------------------------------------

class TestLabelChangeTable:
    def test_square(self):
        A = _make_result(rseed=0)
        B = _make_result(rseed=1)
        result = compare_conditions(A, B)
        ct = label_change_table(result)
        assert ct.shape[0] == ct.shape[1]

    def test_sum_matches_gene_count(self):
        A = _make_result(rseed=0)
        B = _make_result(rseed=1)
        result = compare_conditions(A, B)
        ct = label_change_table(result)
        assert ct.values.sum() == len(result)

    def test_labels_are_valid(self):
        from circ.visualization.classify import _LABEL_ORDER
        A = _make_result(rseed=0)
        B = _make_result(rseed=1)
        result = compare_conditions(A, B)
        ct = label_change_table(result)
        for lbl in ct.index:
            assert lbl in _LABEL_ORDER
        for lbl in ct.columns:
            assert lbl in _LABEL_ORDER
