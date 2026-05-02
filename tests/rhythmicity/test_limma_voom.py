"""Tests for bootjtk.limma_voom — the Python vooma + eBayes preprocessing."""

import numpy as np
import pandas as pd
import pytest

from circ.rhythmicity.limma_voom import (
    _vooma_stats,
    _estimate_prior,
    _posterior_sd,
    _impute_na,
    run_vooma_ebayes,
    run_vooma_vash,
)


# ── fixtures ──────────────────────────────────────────────────────────────


def _make_df(n_genes=10, n_reps=2, period=24, seed=0):
    """Wide-format DataFrame: genes × timepoints, 2 reps per ZT (ZT0..ZT22 × 2)."""
    rng = np.random.default_rng(seed)
    times = list(range(0, 24, 2))  # 12 unique ZTs
    cols = times + [t + 24 for t in times]  # two cycles → 24 columns
    data = 10 + rng.normal(0, 0.5, size=(n_genes, len(cols)))
    return pd.DataFrame(
        data,
        index=[f"gene_{i}" for i in range(n_genes)],
        columns=pd.Index(cols, dtype=float),
    )


# ── _vooma_stats ──────────────────────────────────────────────────────────


class TestVoomaStats:
    def test_output_shape(self):
        df = _make_df(n_genes=5)
        long = _vooma_stats(df, period=24)
        # 5 genes × 12 unique timepoints
        assert len(long) == 5 * 12

    def test_columns_present(self):
        long = _vooma_stats(_make_df(), period=24)
        assert set(long.columns) == {"gene", "time", "mean", "sd_pre", "df", "n"}

    def test_replicate_count(self):
        # Two cycles → each timepoint has n=2 replicates
        long = _vooma_stats(_make_df(), period=24)
        assert (long["n"] == 2).all()

    def test_df_equals_n_minus_1(self):
        long = _vooma_stats(_make_df(), period=24)
        assert (long["df"] == long["n"] - 1).all()

    def test_mean_finite(self):
        long = _vooma_stats(_make_df(), period=24)
        assert long["mean"].notna().all()

    def test_sd_finite_with_replicates(self):
        long = _vooma_stats(_make_df(), period=24)
        assert long["sd_pre"].notna().all()

    def test_sd_nan_without_replicates(self):
        # Single-cycle header → n=1, sd should be NaN
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.normal(10, 1, (4, 12)),
            columns=pd.Index(range(0, 24, 2), dtype=float),
        )
        long = _vooma_stats(df, period=24)
        assert long["sd_pre"].isna().all()

    def test_time_values_in_range(self):
        long = _vooma_stats(_make_df(), period=24)
        assert (long["time"] >= 0).all() and (long["time"] < 24).all()

    def test_known_mean(self):
        # Two columns at ZT0 with values 3 and 5 → mean = 4
        df = pd.DataFrame([[3.0, 5.0]], index=["g"], columns=pd.Index([0.0, 24.0]))
        long = _vooma_stats(df, period=24)
        assert long.loc[0, "mean"] == pytest.approx(4.0)

    def test_known_sd(self):
        # Two columns at ZT0 with values 3 and 5 → sample SD = sqrt(2)
        df = pd.DataFrame([[3.0, 5.0]], index=["g"], columns=pd.Index([0.0, 24.0]))
        long = _vooma_stats(df, period=24)
        assert long.loc[0, "sd_pre"] == pytest.approx(np.sqrt(2.0))


# ── _estimate_prior ───────────────────────────────────────────────────────


class TestEstimatePrior:
    def test_returns_two_floats(self):
        sds = np.abs(np.random.default_rng(0).normal(1, 0.2, 100))
        dfs = np.ones(100)
        d0, s0 = _estimate_prior(sds, dfs)
        assert isinstance(d0, float) and isinstance(s0, float)

    def test_d0_nonnegative(self):
        sds = np.abs(np.random.default_rng(1).normal(1, 0.2, 100))
        dfs = np.ones(100) * 2
        d0, s0 = _estimate_prior(sds, dfs)
        assert d0 >= 0

    def test_s0_positive(self):
        sds = np.abs(np.random.default_rng(2).normal(1, 0.2, 100))
        dfs = np.ones(100) * 2
        d0, s0 = _estimate_prior(sds, dfs)
        assert s0 > 0

    def test_fallback_with_too_few_samples(self):
        # Fewer than 3 valid entries → should not raise
        d0, s0 = _estimate_prior([0.5, 0.6], [1.0, 1.0])
        assert s0 > 0

    def test_ignores_nan_and_zero_sds(self):
        sds = np.array([0.5, 0.0, np.nan, 0.8, 0.6] * 20)
        dfs = np.ones(len(sds)) * 2
        d0, s0 = _estimate_prior(sds, dfs)
        assert np.isfinite(d0) and np.isfinite(s0)


# ── _posterior_sd ─────────────────────────────────────────────────────────


class TestPosteriorSd:
    def test_equals_prior_when_no_data(self):
        # df=0 or NaN sd → should fall back to prior
        result = _posterior_sd(
            np.array([np.nan, np.nan]), np.array([0.0, 0.0]), d0=4.0, s0=1.0
        )
        np.testing.assert_allclose(result, 1.0)

    def test_shrinks_toward_prior(self):
        # High prior (d0 large) should pull posterior toward s0
        d0, s0 = 100.0, 2.0
        result = _posterior_sd(np.array([0.1]), np.array([1.0]), d0=d0, s0=s0)
        assert abs(result[0] - s0) < 0.5

    def test_known_formula(self):
        # sqrt((2*1^2 + 1*1^2) / (2+1)) = sqrt(1) = 1
        result = _posterior_sd(np.array([1.0]), np.array([1.0]), d0=2.0, s0=1.0)
        assert result[0] == pytest.approx(1.0)

    def test_nonnegative_output(self):
        rng = np.random.default_rng(0)
        sds = np.abs(rng.normal(1, 0.3, 50))
        dfs = np.ones(50) * 2
        result = _posterior_sd(sds, dfs, d0=3.0, s0=1.0)
        assert (result >= 0).all()


# ── _impute_na ────────────────────────────────────────────────────────────


class TestImputeNa:
    def test_no_nan_in_output(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.normal(10, 1, (8, 4)), columns=pd.Index([0.0, 1.0, 2.0, 3.0])
        )
        df.iloc[0, 1] = np.nan
        df.iloc[2, :] = np.nan
        result = _impute_na(df)
        assert not result.isna().any().any()

    def test_shape_preserved(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.normal(10, 1, (6, 3)))
        df.iloc[1, 2] = np.nan
        result = _impute_na(df)
        assert result.shape == df.shape

    def test_non_missing_unchanged(self):
        df = pd.DataFrame([[1.0, 2.0, np.nan]], columns=pd.Index([0.0, 1.0, 2.0]))
        result = _impute_na(df)
        assert result.iloc[0, 0] == pytest.approx(1.0)
        assert result.iloc[0, 1] == pytest.approx(2.0)


# ── run_vooma_ebayes (integration) ────────────────────────────────────────


class TestRunVoomaEbayes:
    def test_output_columns(self):
        result = run_vooma_ebayes(_make_df(), period=24)
        assert set(result.columns) == {"ID", "Time", "Mean", "SD", "SDpre", "N"}

    def test_output_row_count(self):
        df = _make_df(n_genes=8)
        result = run_vooma_ebayes(df, period=24)
        assert len(result) == 8 * 12

    def test_sd_nonnegative(self):
        result = run_vooma_ebayes(_make_df(), period=24)
        assert (result["SD"] >= 0).all()

    def test_sd_finite(self):
        result = run_vooma_ebayes(_make_df(), period=24)
        assert result["SD"].notna().all()

    def test_mean_consistent_with_input(self):
        # Gene 0 at ZT0 (columns 0.0 and 24.0): mean should match nanmean
        df = _make_df(n_genes=3)
        result = run_vooma_ebayes(df, period=24)
        zt0_row = result[(result["ID"] == "gene_0") & (result["Time"] == 0.0)]
        expected_mean = float(df.loc["gene_0", [0.0, 24.0]].mean())
        assert float(zt0_row["Mean"].iloc[0]) == pytest.approx(expected_mean, abs=1e-9)

    def test_noisy_gene_gets_higher_sd_than_quiet_gene(self):
        """After shrinkage, a noisier gene should retain a higher posterior SD."""
        rng = np.random.default_rng(42)
        cols = pd.Index([0.0, 24.0, 2.0, 26.0], dtype=float)
        quiet = 10 + rng.normal(0, 0.01, (1, 4))
        noisy = 10 + rng.normal(0, 2.0, (1, 4))
        df = pd.DataFrame(
            np.vstack([quiet, noisy]),
            index=["quiet", "noisy"],
            columns=cols,
        )
        result = run_vooma_ebayes(df, period=24).set_index("ID")
        assert result.loc["noisy", "SD"].mean() > result.loc["quiet", "SD"].mean()

    def test_single_replicate_gets_prior_sd(self):
        """Genes with only one replicate per timepoint receive the prior SD (no NaN)."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.normal(10, 1, (5, 12)),
            columns=pd.Index(range(0, 24, 2), dtype=float),
        )
        result = run_vooma_ebayes(df, period=24)
        assert result["SD"].notna().all()

    def test_rnaseq_flag_accepted(self):
        result = run_vooma_ebayes(_make_df(), period=24, rnaseq=True)
        assert len(result) > 0


# ── run_vooma_vash (integration) ──────────────────────────────────────────


class TestRunVoomaVash:
    def test_output_columns(self):
        result = run_vooma_vash(_make_df(), period=24)
        assert set(result.columns) == {"ID", "Time", "Mean", "SD", "SDpre", "N"}

    def test_output_row_count(self):
        df = _make_df(n_genes=6)
        result = run_vooma_vash(df, period=24)
        assert len(result) == 6 * 12

    def test_handles_na_values(self):
        df = _make_df(n_genes=4)
        df.iloc[0, 2] = np.nan
        df.iloc[2, :] = np.nan
        result = run_vooma_vash(df, period=24)
        assert result["SD"].notna().all()

    def test_sd_nonnegative(self):
        df = _make_df(n_genes=5)
        df.iloc[1, 3] = np.nan
        result = run_vooma_vash(df, period=24)
        assert (result["SD"] >= 0).all()


# ── write_limma_outputs round-trip ────────────────────────────────────────


class TestWriteLimmaOutputsRoundTrip:
    def test_round_trip_produces_correct_files(self, tmp_path):
        from circ.rhythmicity.limma_preprocess import write_limma_outputs

        df = _make_df(n_genes=4)
        long_df = run_vooma_ebayes(df, period=24)
        pref = str(tmp_path / "out")
        write_limma_outputs(long_df, pref, "postLimma")
        means_path = tmp_path / "out_Means_postLimma.txt"
        assert means_path.exists()
        wide = pd.read_table(str(means_path), index_col="ID")
        assert wide.shape == (4, 12)  # 4 genes × 12 unique timepoints

    def test_sds_file_values_positive(self, tmp_path):
        from circ.rhythmicity.limma_preprocess import write_limma_outputs

        df = _make_df(n_genes=4)
        long_df = run_vooma_ebayes(df, period=24)
        write_limma_outputs(long_df, str(tmp_path / "out"), "postLimma")
        sds = pd.read_table(str(tmp_path / "out_Sds_postLimma.txt"), index_col="ID")
        assert (sds.values > 0).all()
