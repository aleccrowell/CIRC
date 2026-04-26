"""Tests for circ.visualization.compare — condition comparison plots."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes

from circ.compare import compare_conditions
from circ.visualization.compare import (
    rhythmicity_shift_scatter,
    phase_shift_histogram,
    label_transition_heatmap,
    delta_tau_volcano,
    comparison_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(rseed=0, with_uncertainty=False, n=80):
    rng = np.random.default_rng(rseed)
    idx = pd.Index([f'gene_{i:04d}' for i in range(n)], name='#')
    tau   = rng.uniform(0.0, 1.0, n)
    emp_p = np.where(tau > 0.5, rng.uniform(0.0, 0.05, n), rng.uniform(0.05, 1.0, n))
    pirs  = rng.uniform(0.0, 2.0, n)
    phase = rng.uniform(0.0, 24.0, n)
    labels = np.where(
        (tau > 0.5) & (emp_p < 0.05), 'rhythmic',
        np.where(pirs < np.percentile(pirs, 50), 'constitutive', 'variable'),
    )
    df = pd.DataFrame({
        'tau_mean': tau, 'emp_p': emp_p, 'pirs_score': pirs,
        'phase_mean': phase, 'label': labels,
    }, index=idx)
    if with_uncertainty:
        df['tau_std']   = rng.uniform(0.05, 0.2, n)
        df['phase_std'] = rng.uniform(0.5, 2.0, n)
        df['n_boots']   = 50
    return df


@pytest.fixture
def comp_basic():
    A = _make_result(rseed=0)
    B = _make_result(rseed=1)
    return compare_conditions(A, B)


@pytest.fixture
def comp_with_uncertainty():
    A = _make_result(rseed=0, with_uncertainty=True)
    B = _make_result(rseed=1, with_uncertainty=True)
    return compare_conditions(A, B)


# ---------------------------------------------------------------------------
# rhythmicity_shift_scatter
# ---------------------------------------------------------------------------

class TestRhythmicityShiftScatter:
    def test_returns_axes(self, comp_basic):
        _, ax = plt.subplots()
        result = rhythmicity_shift_scatter(comp_basic, ax=ax)
        assert isinstance(result, matplotlib.axes.Axes)
        plt.close()

    def test_creates_axes_if_none(self, comp_basic):
        result = rhythmicity_shift_scatter(comp_basic)
        assert isinstance(result, matplotlib.axes.Axes)
        plt.close()

    def test_with_padj(self, comp_with_uncertainty):
        _, ax = plt.subplots()
        result = rhythmicity_shift_scatter(comp_with_uncertainty, ax=ax)
        assert isinstance(result, matplotlib.axes.Axes)
        plt.close()


# ---------------------------------------------------------------------------
# phase_shift_histogram
# ---------------------------------------------------------------------------

class TestPhaseShiftHistogram:
    def test_returns_axes(self, comp_basic):
        _, ax = plt.subplots()
        result = phase_shift_histogram(comp_basic, ax=ax)
        assert isinstance(result, matplotlib.axes.Axes)
        plt.close()

    def test_no_phase_data(self):
        A = _make_result(rseed=0).drop(columns=['phase_mean'])
        B = _make_result(rseed=1).drop(columns=['phase_mean'])
        comp = compare_conditions(A, B)
        _, ax = plt.subplots()
        result = phase_shift_histogram(comp, ax=ax)
        assert isinstance(result, matplotlib.axes.Axes)
        plt.close()

    def test_no_rhythmic_genes(self):
        # Force all genes to be non-rhythmic so delta_phase is all NaN
        A = _make_result(rseed=0)
        A['emp_p'] = 1.0   # nothing passes emp_p < 0.05
        B = _make_result(rseed=1)
        B['emp_p'] = 1.0
        comp = compare_conditions(A, B)
        _, ax = plt.subplots()
        result = phase_shift_histogram(comp, ax=ax)
        assert isinstance(result, matplotlib.axes.Axes)
        plt.close()


# ---------------------------------------------------------------------------
# label_transition_heatmap
# ---------------------------------------------------------------------------

class TestLabelTransitionHeatmap:
    def test_returns_axes(self, comp_basic):
        _, ax = plt.subplots()
        result = label_transition_heatmap(comp_basic, ax=ax)
        assert isinstance(result, matplotlib.axes.Axes)
        plt.close()


# ---------------------------------------------------------------------------
# delta_tau_volcano
# ---------------------------------------------------------------------------

class TestDeltaTauVolcano:
    def test_raises_without_padj(self, comp_basic):
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match='tau_padj'):
            delta_tau_volcano(comp_basic, ax=ax)
        plt.close()

    def test_returns_axes_with_padj(self, comp_with_uncertainty):
        _, ax = plt.subplots()
        result = delta_tau_volcano(comp_with_uncertainty, ax=ax)
        assert isinstance(result, matplotlib.axes.Axes)
        plt.close()


# ---------------------------------------------------------------------------
# comparison_summary
# ---------------------------------------------------------------------------

class TestComparisonSummary:
    def test_returns_figure(self, comp_basic):
        fig = comparison_summary(comp_basic)
        assert fig is not None
        plt.close('all')

    def test_returns_figure_with_uncertainty(self, comp_with_uncertainty):
        fig = comparison_summary(comp_with_uncertainty)
        assert fig is not None
        plt.close('all')

    def test_saves_to_file(self, comp_basic, tmp_path):
        out = str(tmp_path / 'test_summary.png')
        fig = comparison_summary(comp_basic, outpath=out)
        assert fig is not None
        import os
        assert os.path.exists(out)
        plt.close('all')
