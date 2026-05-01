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
    gene_profile,
    expression_heatmap,
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


@pytest.fixture
def expression_df(minimal_clf):
    """Tiny expression matrix with ZT columns aligned to minimal_clf index."""
    rng = np.random.default_rng(5)
    n = len(minimal_clf)
    cols = [f'ZT{h:02d}_{r}' for h in [0, 4, 8, 12, 16, 20] for r in [1, 2]]
    return pd.DataFrame(rng.normal(5, 1, (n, len(cols))), index=minimal_clf.index, columns=cols)


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
        #         + top_constitutive_candidates + threshold_sensitivity = 5
        assert len(fig.axes) == 5

    def test_returns_figure_full(self, full_clf):
        fig = classification_summary(full_clf)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 5

    def test_saves_to_file(self, full_clf, tmp_path):
        out = str(tmp_path / 'summary.png')
        classification_summary(full_clf, outpath=out)
        import os
        assert os.path.exists(out)

    def test_minimal_has_correct_panel_count(self, minimal_clf):
        fig = classification_summary(minimal_clf)
        # label_distribution + pirs_vs_tau + pirs_score_distribution
        # + top_constitutive_candidates + threshold_sensitivity = 5
        assert len(fig.axes) == 5

    def test_with_emp_p_adds_two_panels(self, minimal_clf):
        df = minimal_clf.copy()
        rng = np.random.default_rng(2)
        df['emp_p'] = rng.uniform(0, 1, len(df))
        fig = classification_summary(df)
        # 5 base + 2 (volcano + tau_pval_scatter) = 7
        assert len(fig.axes) == 7


# ---------------------------------------------------------------------------
# label_distribution xlim
# ---------------------------------------------------------------------------

class TestLabelDistributionXlim:
    def test_xlim_applied(self, minimal_clf):
        ax = label_distribution(minimal_clf, xlim=500)
        assert ax.get_xlim()[1] == pytest.approx(500)

    def test_xlim_none_leaves_default(self, minimal_clf):
        ax = label_distribution(minimal_clf)
        assert ax.get_xlim()[1] != pytest.approx(500)


# ---------------------------------------------------------------------------
# gene_profile
# ---------------------------------------------------------------------------

class TestGeneProfile:
    def test_returns_axes(self, expression_df, minimal_clf):
        gene = expression_df.index[0]
        ax = gene_profile(expression_df, gene, minimal_clf)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_accepts_explicit_ax(self, expression_df, minimal_clf):
        _, ax = plt.subplots()
        gene = expression_df.index[0]
        returned = gene_profile(expression_df, gene, minimal_clf, ax=ax)
        assert returned is ax

    def test_title_contains_gene_id(self, expression_df, minimal_clf):
        gene = expression_df.index[3]
        ax = gene_profile(expression_df, gene, minimal_clf)
        assert gene in ax.get_title()

    def test_custom_title(self, expression_df, minimal_clf):
        gene = expression_df.index[0]
        ax = gene_profile(expression_df, gene, minimal_clf, title="My title")
        assert ax.get_title() == "My title"

    def test_raises_on_missing_gene(self, expression_df, minimal_clf):
        with pytest.raises(ValueError, match='not found'):
            gene_profile(expression_df, 'nonexistent_gene', minimal_clf)

    def test_works_without_classifications(self, expression_df):
        gene = expression_df.index[0]
        ax = gene_profile(expression_df, gene)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_scatter_and_line_drawn(self, expression_df, minimal_clf):
        gene = expression_df.index[0]
        ax = gene_profile(expression_df, gene, minimal_clf)
        assert len(ax.collections) >= 1   # scatter
        assert len(ax.lines) >= 1         # mean line

    def test_color_from_label(self, expression_df, minimal_clf):
        gene = minimal_clf[minimal_clf['label'] == 'rhythmic'].index[0]
        ax = gene_profile(expression_df, gene, minimal_clf)
        scatter_color = ax.collections[0].get_facecolors()[0][:3]  # RGB only; alpha set by scatter
        expected = plt.matplotlib.colors.to_rgb(LABEL_COLORS['rhythmic'])
        np.testing.assert_allclose(scatter_color, expected, atol=0.01)


# ---------------------------------------------------------------------------
# expression_heatmap
# ---------------------------------------------------------------------------

class TestExpressionHeatmap:
    def test_returns_axes(self, expression_df, minimal_clf):
        ax = expression_heatmap(expression_df, minimal_clf)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_works_without_classifications(self, expression_df):
        ax = expression_heatmap(expression_df)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_accepts_explicit_ax(self, expression_df, minimal_clf):
        _, ax = plt.subplots()
        returned = expression_heatmap(expression_df, minimal_clf, ax=ax)
        assert returned is ax

    def test_heatmap_drawn(self, expression_df, minimal_clf):
        ax = expression_heatmap(expression_df, minimal_clf)
        # imshow produces an AxesImage in ax.images
        assert len(ax.images) >= 1

    def test_n_per_label_limits_rows(self, expression_df, minimal_clf):
        ax = expression_heatmap(expression_df, minimal_clf, n_per_label=3)
        img = ax.images[0]
        n_labels = minimal_clf['label'].nunique()
        assert img.get_array().shape[0] <= 3 * n_labels

    def test_colorbar_present(self, expression_df, minimal_clf):
        ax = expression_heatmap(expression_df, minimal_clf, colorbar=True)
        fig = ax.get_figure()
        # colorbar creates an extra axes in the figure
        assert len(fig.axes) >= 2

    def test_no_colorbar(self, expression_df, minimal_clf):
        _, ax = plt.subplots()
        expression_heatmap(expression_df, minimal_clf, colorbar=False)
        fig = ax.get_figure()
        # label strip adds one extra axes, but no colorbar
        assert len(fig.axes) <= 2

    def test_custom_title(self, expression_df, minimal_clf):
        ax = expression_heatmap(expression_df, minimal_clf, title="My heatmap")
        assert ax.get_title() == "My heatmap"

    def test_raises_on_no_zt_columns(self, minimal_clf):
        bad_expr = pd.DataFrame(
            np.ones((10, 4)),
            index=minimal_clf.index[:10],
            columns=['A', 'B', 'C', 'D'],
        )
        with pytest.raises(ValueError, match='ZT/CT'):
            expression_heatmap(bad_expr, minimal_clf)
