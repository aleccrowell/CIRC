"""Tests for circ.visualization.classify — expression classification plots."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes

from circ.visualization.classify import (
    LABEL_COLORS,
    label_distribution,
    pirs_vs_tau,
    volcano,
    pirs_score_distribution,
    tau_pval_scatter,
    pirs_pval_scatter,
    slope_pval_scatter,
    slope_vs_rhythm,
    phase_wheel,
    period_distribution,
    classification_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_clf():
    """Minimal classification DataFrame: only the columns always present."""
    rng = np.random.default_rng(0)
    n = 60
    labels = (
        ['constitutive'] * 15 + ['rhythmic'] * 15 +
        ['variable'] * 15 + ['noisy_rhythmic'] * 15
    )
    return pd.DataFrame({
        'pirs_score': np.abs(rng.normal(0, 1, n)),
        'tau_mean':   rng.uniform(0, 1, n),
        'label':      labels,
    }, index=[f'g{i}' for i in range(n)])


@pytest.fixture
def full_clf(minimal_clf):
    """Classification DataFrame with all optional columns present."""
    rng = np.random.default_rng(1)
    n = len(minimal_clf)
    df = minimal_clf.copy()
    df['emp_p']           = rng.uniform(0, 1, n)
    df['pval']            = rng.uniform(0, 1, n)
    df['pval_bh']         = rng.uniform(0, 1, n)
    df['slope_pval']      = rng.uniform(0, 1, n)
    df['slope_pval_bh']   = rng.uniform(0, 1, n)
    df['phase_mean']      = rng.uniform(0, 24, n)
    df['period_mean']     = rng.normal(24, 1, n)
    return df


@pytest.fixture
def with_linear(full_clf):
    """Adds linear genes so all five labels are present."""
    extra = full_clf.iloc[:10].copy()
    extra.index = [f'lin{i}' for i in range(10)]
    extra['label'] = 'linear'
    return pd.concat([full_clf, extra])


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


# ---------------------------------------------------------------------------
# LABEL_COLORS
# ---------------------------------------------------------------------------

def test_label_colors_has_all_labels():
    for lbl in ('constitutive', 'rhythmic', 'linear', 'variable',
                'noisy_rhythmic', 'unclassified'):
        assert lbl in LABEL_COLORS
        assert LABEL_COLORS[lbl].startswith('#')


# ---------------------------------------------------------------------------
# label_distribution
# ---------------------------------------------------------------------------

class TestLabelDistribution:
    def test_returns_axes(self, minimal_clf):
        ax = label_distribution(minimal_clf)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_bar_count_matches_labels(self, minimal_clf):
        ax = label_distribution(minimal_clf)
        assert len(ax.patches) == minimal_clf['label'].nunique()

    def test_accepts_explicit_ax(self, minimal_clf):
        _, ax = plt.subplots()
        returned = label_distribution(minimal_clf, ax=ax)
        assert returned is ax

    def test_all_five_labels_shown(self, with_linear):
        ax = label_distribution(with_linear)
        assert len(ax.patches) == 5


# ---------------------------------------------------------------------------
# pirs_vs_tau
# ---------------------------------------------------------------------------

class TestPirsVsTau:
    def test_returns_axes(self, minimal_clf):
        ax = pirs_vs_tau(minimal_clf)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_decision_lines_drawn(self, minimal_clf):
        ax = pirs_vs_tau(minimal_clf)
        lines = [l for l in ax.lines if l.get_xdata() is not None]
        assert len(lines) >= 2

    def test_accepts_explicit_ax(self, minimal_clf):
        _, ax = plt.subplots()
        returned = pirs_vs_tau(minimal_clf, ax=ax)
        assert returned is ax


# ---------------------------------------------------------------------------
# volcano
# ---------------------------------------------------------------------------

class TestVolcano:
    def test_returns_axes(self, full_clf):
        ax = volcano(full_clf)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_raises_without_emp_p(self, minimal_clf):
        with pytest.raises(ValueError, match='emp_p'):
            volcano(minimal_clf)

    def test_accepts_explicit_ax(self, full_clf):
        _, ax = plt.subplots()
        returned = volcano(full_clf, ax=ax)
        assert returned is ax


# ---------------------------------------------------------------------------
# pirs_score_distribution
# ---------------------------------------------------------------------------

class TestPirsScoreDistribution:
    def test_returns_axes(self, minimal_clf):
        ax = pirs_score_distribution(minimal_clf)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_vertical_line_present(self, minimal_clf):
        ax = pirs_score_distribution(minimal_clf)
        vlines = [l for l in ax.lines]
        assert len(vlines) >= 1


# ---------------------------------------------------------------------------
# tau_pval_scatter
# ---------------------------------------------------------------------------

class TestTauPvalScatter:
    def test_returns_axes(self, full_clf):
        ax = tau_pval_scatter(full_clf)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_raises_without_emp_p(self, minimal_clf):
        with pytest.raises(ValueError, match='emp_p'):
            tau_pval_scatter(minimal_clf)


# ---------------------------------------------------------------------------
# pirs_pval_scatter
# ---------------------------------------------------------------------------

class TestPirsPvalScatter:
    def test_returns_axes(self, full_clf):
        ax = pirs_pval_scatter(full_clf)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_raises_without_pval(self, minimal_clf):
        with pytest.raises(ValueError, match='pval'):
            pirs_pval_scatter(minimal_clf)

    def test_prefers_bh_corrected(self, full_clf):
        ax = pirs_pval_scatter(full_clf)
        assert 'pval_bh' in ax.get_ylabel()


# ---------------------------------------------------------------------------
# slope_pval_scatter
# ---------------------------------------------------------------------------

class TestSlopePvalScatter:
    def test_returns_axes(self, full_clf):
        ax = slope_pval_scatter(full_clf)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_raises_without_slope_pval(self, minimal_clf):
        with pytest.raises(ValueError, match='slope_pval'):
            slope_pval_scatter(minimal_clf)


# ---------------------------------------------------------------------------
# slope_vs_rhythm
# ---------------------------------------------------------------------------

class TestSlopeVsRhythm:
    def test_returns_axes(self, full_clf):
        ax = slope_vs_rhythm(full_clf)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_raises_without_slope_pval(self, full_clf):
        df = full_clf.drop(columns=['slope_pval', 'slope_pval_bh'])
        with pytest.raises(ValueError, match='slope_pval'):
            slope_vs_rhythm(df)

    def test_raises_without_emp_p(self, full_clf):
        df = full_clf.drop(columns=['emp_p'])
        with pytest.raises(ValueError, match='emp_p'):
            slope_vs_rhythm(df)


# ---------------------------------------------------------------------------
# phase_wheel
# ---------------------------------------------------------------------------

class TestPhaseWheel:
    def test_returns_polar_axes(self, full_clf):
        ax = phase_wheel(full_clf)
        assert ax.name == 'polar'

    def test_raises_without_phase_mean(self, minimal_clf):
        with pytest.raises(ValueError, match='phase_mean'):
            phase_wheel(minimal_clf)

    def test_accepts_explicit_polar_ax(self, full_clf):
        _, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        returned = phase_wheel(full_clf, ax=ax)
        assert returned is ax

    def test_fallback_to_all_genes_when_no_rhythmic(self, full_clf):
        """When no genes carry a rhythmic label, fall back to all genes."""
        df = full_clf.copy()
        df['label'] = 'constitutive'  # strip all rhythmic labels
        ax = phase_wheel(df)
        assert ax.name == 'polar'
        assert 'all genes' in ax.get_title()


# ---------------------------------------------------------------------------
# period_distribution
# ---------------------------------------------------------------------------

class TestPeriodDistribution:
    def test_returns_axes(self, full_clf):
        ax = period_distribution(full_clf)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_raises_without_period_mean(self, minimal_clf):
        with pytest.raises(ValueError, match='period_mean'):
            period_distribution(minimal_clf)

    def test_reference_line_present(self, full_clf):
        ax = period_distribution(full_clf)
        vlines = [l for l in ax.lines]
        assert len(vlines) >= 1

    def test_fallback_to_all_genes_when_no_rhythmic(self, full_clf):
        """When no genes carry a rhythmic label, fall back to all genes."""
        df = full_clf.copy()
        df['label'] = 'constitutive'
        ax = period_distribution(df)
        assert isinstance(ax, matplotlib.axes.Axes)
        assert 'all genes' in ax.get_title()
        assert len(ax.patches) > 0  # histogram bars drawn

    def test_constant_period_shows_bars(self, full_clf):
        """Constant period_mean (all 24 h) must still produce visible bars."""
        df = full_clf.copy()
        df['period_mean'] = 24.0
        ax = period_distribution(df)
        assert len(ax.patches) > 0


# ---------------------------------------------------------------------------
# classification_summary
# ---------------------------------------------------------------------------

class TestClassificationSummary:
    def test_returns_figure_minimal(self, minimal_clf):
        fig = classification_summary(minimal_clf)
        assert isinstance(fig, plt.Figure)
        # always: label_distribution + pirs_vs_tau + pirs_score_distribution
        #         + top_constitutive_candidates = 4
        assert len(fig.axes) == 4

    def test_returns_figure_full(self, full_clf):
        fig = classification_summary(full_clf)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 4

    def test_saves_to_file(self, full_clf, tmp_path):
        out = str(tmp_path / 'summary.png')
        classification_summary(full_clf, outpath=out)
        import os
        assert os.path.exists(out)

    def test_minimal_has_correct_panel_count(self, minimal_clf):
        fig = classification_summary(minimal_clf)
        # label_distribution + pirs_vs_tau + pirs_score_distribution
        # + top_constitutive_candidates = 4
        assert len(fig.axes) == 4

    def test_with_emp_p_adds_two_panels(self, minimal_clf):
        df = minimal_clf.copy()
        rng = np.random.default_rng(2)
        df['emp_p'] = rng.uniform(0, 1, len(df))
        fig = classification_summary(df)
        # 4 base + 2 (volcano + tau_pval_scatter) = 6
        assert len(fig.axes) == 6
