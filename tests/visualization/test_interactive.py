"""Tests for circ.visualization.interactive — Plotly figure construction."""
import re

import numpy as np
import pandas as pd
import pytest

plotly = pytest.importorskip('plotly')
import plotly.graph_objects as go

from circ.visualization.interactive.classify import (
    label_distribution,
    pirs_vs_tau,
    volcano,
    pirs_score_distribution,
    tau_pval_scatter,
    phase_wheel,
    period_distribution,
    classification_summary,
)
from circ.visualization.interactive.benchmarks import (
    classification_pr,
    classification_roc,
)
from circ.visualization.interactive.compare import (
    rhythmicity_shift_scatter,
    delta_tau_volcano,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_df():
    rng = np.random.default_rng(0)
    n = 100
    labels = ['constitutive'] * 25 + ['rhythmic'] * 25 + ['variable'] * 25 + ['noisy_rhythmic'] * 25
    return pd.DataFrame({
        'pirs_score':  np.abs(rng.normal(0, 1, n)),
        'tau_mean':    rng.uniform(0, 1, n),
        'emp_p':       rng.uniform(0, 1, n),
        'pval':        rng.uniform(0, 1, n),
        'pval_bh':     rng.uniform(0, 1, n),
        'slope_pval':  rng.uniform(0, 1, n),
        'phase_mean':  rng.uniform(0, 24, n),
        'period_mean': rng.uniform(20, 28, n),
        'label':       labels,
    }, index=[f'gene_{i}' for i in range(n)])


@pytest.fixture
def true_classes(clf_df):
    rng = np.random.default_rng(1)
    n = len(clf_df)
    return pd.DataFrame({
        'Const':     rng.integers(0, 2, n),
        'Circadian': rng.integers(0, 2, n),
        'Linear':    rng.integers(0, 2, n),
    }, index=clf_df.index)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_figure(obj):
    return isinstance(obj, go.Figure)


def _scatter_traces(fig):
    return [t for t in fig.data if isinstance(t, go.Scatter)]


def _has_gene_ids(fig):
    """At least one Scatter trace carries gene IDs in its text field."""
    return any(
        isinstance(t, go.Scatter) and t.text and len(t.text) > 0
        for t in fig.data
    )


def _n_shapes(fig):
    return len(fig.layout.shapes) if fig.layout.shapes else 0


# ---------------------------------------------------------------------------
# label_distribution
# ---------------------------------------------------------------------------

class TestLabelDistribution:
    def test_returns_figure(self, clf_df):
        assert _is_figure(label_distribution(clf_df))

    def test_one_bar_for_present_labels(self, clf_df):
        fig = label_distribution(clf_df)
        assert len(fig.data) == 1
        bar = fig.data[0]
        assert len(bar.y) == clf_df['label'].nunique()

    def test_custom_title(self, clf_df):
        fig = label_distribution(clf_df, title='My title')
        assert fig.layout.title.text == 'My title'


# ---------------------------------------------------------------------------
# pirs_vs_tau
# ---------------------------------------------------------------------------

class TestPirsVsTau:
    def test_returns_figure(self, clf_df):
        assert _is_figure(pirs_vs_tau(clf_df))

    def test_one_trace_per_label(self, clf_df):
        fig = pirs_vs_tau(clf_df)
        n_labels = clf_df['label'].nunique()
        assert len(_scatter_traces(fig)) == n_labels

    def test_gene_ids_in_hover(self, clf_df):
        fig = pirs_vs_tau(clf_df)
        assert _has_gene_ids(fig)

    def test_threshold_shapes_added(self, clf_df):
        fig = pirs_vs_tau(clf_df)
        assert _n_shapes(fig) >= 2


# ---------------------------------------------------------------------------
# volcano
# ---------------------------------------------------------------------------

class TestVolcano:
    def test_returns_figure(self, clf_df):
        assert _is_figure(volcano(clf_df))

    def test_raises_without_emp_p(self, clf_df):
        with pytest.raises(ValueError, match='emp_p'):
            volcano(clf_df.drop(columns=['emp_p']))

    def test_gene_ids_in_hover(self, clf_df):
        fig = volcano(clf_df)
        assert _has_gene_ids(fig)

    def test_threshold_shapes_added(self, clf_df):
        fig = volcano(clf_df)
        assert _n_shapes(fig) >= 2


# ---------------------------------------------------------------------------
# pirs_score_distribution
# ---------------------------------------------------------------------------

class TestPirsScoreDistribution:
    def test_returns_figure(self, clf_df):
        assert _is_figure(pirs_score_distribution(clf_df))

    def test_one_histogram_per_label(self, clf_df):
        fig = pirs_score_distribution(clf_df)
        hists = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(hists) == clf_df['label'].nunique()

    def test_threshold_shape_added(self, clf_df):
        fig = pirs_score_distribution(clf_df)
        assert _n_shapes(fig) >= 1


# ---------------------------------------------------------------------------
# tau_pval_scatter
# ---------------------------------------------------------------------------

class TestTauPvalScatter:
    def test_returns_figure(self, clf_df):
        assert _is_figure(tau_pval_scatter(clf_df))

    def test_raises_without_emp_p(self, clf_df):
        with pytest.raises(ValueError, match='emp_p'):
            tau_pval_scatter(clf_df.drop(columns=['emp_p']))

    def test_gene_ids_in_hover(self, clf_df):
        fig = tau_pval_scatter(clf_df)
        assert _has_gene_ids(fig)


# ---------------------------------------------------------------------------
# phase_wheel
# ---------------------------------------------------------------------------

class TestPhaseWheel:
    def test_returns_figure(self, clf_df):
        assert _is_figure(phase_wheel(clf_df))

    def test_raises_without_phase_mean(self, clf_df):
        with pytest.raises(ValueError, match='phase_mean'):
            phase_wheel(clf_df.drop(columns=['phase_mean']))

    def test_barpolar_trace_present(self, clf_df):
        fig = phase_wheel(clf_df)
        assert any(isinstance(t, go.Barpolar) for t in fig.data)

    def test_12_bins(self, clf_df):
        fig = phase_wheel(clf_df)
        bar = next(t for t in fig.data if isinstance(t, go.Barpolar))
        assert len(bar.r) == 12

    def test_fallback_to_all_genes_when_no_rhythmic(self, clf_df):
        df = clf_df.copy()
        df['label'] = 'constitutive'
        fig = phase_wheel(df)
        assert 'all genes' in fig.layout.title.text


# ---------------------------------------------------------------------------
# period_distribution
# ---------------------------------------------------------------------------

class TestPeriodDistribution:
    def test_returns_figure(self, clf_df):
        assert _is_figure(period_distribution(clf_df))

    def test_raises_without_period_mean(self, clf_df):
        with pytest.raises(ValueError, match='period_mean'):
            period_distribution(clf_df.drop(columns=['period_mean']))

    def test_reference_line_added(self, clf_df):
        fig = period_distribution(clf_df)
        assert _n_shapes(fig) >= 1


# ---------------------------------------------------------------------------
# classification_summary
# ---------------------------------------------------------------------------

class TestClassificationSummary:
    def test_returns_figure(self, clf_df):
        assert _is_figure(classification_summary(clf_df))

    def test_has_multiple_traces(self, clf_df):
        fig = classification_summary(clf_df)
        assert len(fig.data) > 3

    def test_minimal_columns_no_crash(self):
        rng = np.random.default_rng(7)
        n = 40
        df = pd.DataFrame({
            'pirs_score': rng.uniform(0, 1, n),
            'tau_mean':   rng.uniform(0, 1, n),
            'label':      ['constitutive'] * 20 + ['variable'] * 20,
        }, index=[f'g{i}' for i in range(n)])
        fig = classification_summary(df)
        assert _is_figure(fig)


# ---------------------------------------------------------------------------
# classification_pr
# ---------------------------------------------------------------------------

class TestClassificationPr:
    def test_returns_figure(self, clf_df, true_classes):
        assert _is_figure(classification_pr(clf_df, true_classes))

    def test_ap_in_trace_name(self, clf_df, true_classes):
        fig = classification_pr(clf_df, true_classes)
        assert any('AP=' in t.name for t in fig.data if t.name)

    def test_threshold_in_hover(self, clf_df, true_classes):
        fig = classification_pr(clf_df, true_classes)
        trace = fig.data[0]
        assert trace.customdata is not None

    def test_empty_intersection_no_crash(self, clf_df, true_classes):
        tc = true_classes.copy()
        tc.index = [f'other_{i}' for i in range(len(tc))]
        fig = classification_pr(clf_df, tc)
        assert _is_figure(fig)


# ---------------------------------------------------------------------------
# classification_roc
# ---------------------------------------------------------------------------

class TestClassificationRoc:
    def test_returns_figure(self, clf_df, true_classes):
        assert _is_figure(classification_roc(clf_df, true_classes))

    def test_auc_in_trace_names(self, clf_df, true_classes):
        fig = classification_roc(clf_df, true_classes)
        auc_labels = [t.name for t in fig.data if t.name and 'AUC=' in t.name]
        assert len(auc_labels) >= 1

    def test_diagonal_trace_present(self, clf_df, true_classes):
        fig = classification_roc(clf_df, true_classes)
        assert any('random' in (t.name or '') for t in fig.data)

    def test_explicit_tasks(self, clf_df, true_classes):
        fig = classification_roc(clf_df, true_classes,
                                 tasks=[('pirs_score', 'Const', True)])
        # one method trace + diagonal
        assert len(fig.data) == 2

    def test_perfect_score_auc_near_one(self):
        n = 100
        clf = pd.DataFrame(
            {'pirs_score': np.array([0.001] * 25 + [0.99] * 75)},
            index=[f'g{i}' for i in range(n)],
        )
        truth = pd.DataFrame(
            {'Const': np.array([1] * 25 + [0] * 75)},
            index=clf.index,
        )
        fig = classification_roc(clf, truth, tasks=[('pirs_score', 'Const', True)])
        auc_match = re.search(r'AUC=([0-9.]+)', fig.data[0].name)
        assert auc_match is not None
        assert float(auc_match.group(1)) > 0.99

    def test_empty_intersection_no_crash(self, clf_df, true_classes):
        tc = true_classes.copy()
        tc.index = [f'other_{i}' for i in range(len(tc))]
        fig = classification_roc(clf_df, tc)
        assert _is_figure(fig)


# ---------------------------------------------------------------------------
# Fixtures shared by compare tests
# ---------------------------------------------------------------------------

def _make_compare_result(rseed=0, with_uncertainty=False, n=80):
    from circ.compare import compare_conditions
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
    from circ.compare import compare_conditions
    return compare_conditions(_make_compare_result(0), _make_compare_result(1))


@pytest.fixture
def comp_with_uncertainty():
    from circ.compare import compare_conditions
    return compare_conditions(
        _make_compare_result(0, with_uncertainty=True),
        _make_compare_result(1, with_uncertainty=True),
    )


# ---------------------------------------------------------------------------
# rhythmicity_shift_scatter (interactive)
# ---------------------------------------------------------------------------

class TestInteractiveRhythmicityShiftScatter:
    def test_returns_figure(self, comp_basic):
        fig = rhythmicity_shift_scatter(comp_basic)
        assert _is_figure(fig)

    def test_gene_ids_in_hover(self, comp_basic):
        fig = rhythmicity_shift_scatter(comp_basic)
        all_text = [t for trace in fig.data if hasattr(trace, 'text') and trace.text for t in trace.text]
        assert len(all_text) > 0

    def test_diagonal_trace_present(self, comp_basic):
        fig = rhythmicity_shift_scatter(comp_basic)
        line_traces = [t for t in fig.data if t.mode == 'lines']
        assert len(line_traces) >= 1

    def test_with_padj_splits_sig_nonsig(self, comp_with_uncertainty):
        fig = rhythmicity_shift_scatter(comp_with_uncertainty)
        names = [t.name for t in fig.data]
        assert any('FDR<' in n for n in names)

    def test_custom_title(self, comp_basic):
        fig = rhythmicity_shift_scatter(comp_basic, title='My title')
        assert fig.layout.title.text == 'My title'


# ---------------------------------------------------------------------------
# delta_tau_volcano (interactive)
# ---------------------------------------------------------------------------

class TestInteractiveDeltaTauVolcano:
    def test_returns_figure(self, comp_with_uncertainty):
        fig = delta_tau_volcano(comp_with_uncertainty)
        assert _is_figure(fig)

    def test_gene_ids_in_hover(self, comp_with_uncertainty):
        fig = delta_tau_volcano(comp_with_uncertainty)
        all_text = [t for trace in fig.data if hasattr(trace, 'text') and trace.text for t in trace.text]
        assert len(all_text) > 0

    def test_raises_without_tau_padj(self, comp_basic):
        assert 'tau_padj' not in comp_basic.columns
        with pytest.raises(ValueError, match='tau_padj'):
            delta_tau_volcano(comp_basic)

    def test_hline_present(self, comp_with_uncertainty):
        fig = delta_tau_volcano(comp_with_uncertainty)
        assert len(fig.layout.shapes) >= 1


# ---------------------------------------------------------------------------
# expression_heatmap (interactive)
# ---------------------------------------------------------------------------

from circ.visualization.interactive.classify import expression_heatmap as iviz_expression_heatmap


@pytest.fixture
def expr_df(clf_df):
    """Tiny expression matrix aligned to clf_df."""
    rng = np.random.default_rng(10)
    n = len(clf_df)
    cols = [f'ZT{h:02d}_{r}' for h in [0, 4, 8, 12, 16, 20] for r in [1, 2]]
    return pd.DataFrame(rng.normal(5, 1, (n, len(cols))), index=clf_df.index, columns=cols)


class TestInteractiveExpressionHeatmap:
    def test_returns_figure(self, expr_df, clf_df):
        fig = iviz_expression_heatmap(expr_df, clf_df)
        assert _is_figure(fig)

    def test_heatmap_trace_present(self, expr_df, clf_df):
        import plotly.graph_objects as go
        fig = iviz_expression_heatmap(expr_df, clf_df)
        assert any(isinstance(t, go.Heatmap) for t in fig.data)

    def test_gene_ids_in_y(self, expr_df, clf_df):
        fig = iviz_expression_heatmap(expr_df, clf_df)
        heatmap = next(t for t in fig.data if isinstance(t, go.Heatmap))
        assert len(heatmap.y) > 0

    def test_works_without_classifications(self, expr_df):
        fig = iviz_expression_heatmap(expr_df)
        assert _is_figure(fig)

    def test_custom_title(self, expr_df, clf_df):
        fig = iviz_expression_heatmap(expr_df, clf_df, title='My title')
        assert fig.layout.title.text == 'My title'
