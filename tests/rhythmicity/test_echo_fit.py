"""Tests for circ.rhythmicity.echo_fit.EchoFitter."""

import numpy as np
import pandas as pd
import pytest

from circ.rhythmicity.echo_fit import EchoFitter, _ECHO_OUTPUT_COLS, _parse_timepoints


# ---------------------------------------------------------------------------
# Helpers — synthetic oscillation data
# ---------------------------------------------------------------------------


def _make_oscillation_df(
    gamma_norm: float,
    n_genes: int = 5,
    A: float = 2.0,
    baseline: float = 5.0,
    noise: float = 0.01,
    seed: int = 0,
    zt_prefix: str = "ZT",
    tpoint_space: int = 2,
    n_timepoints: int = 12,
    n_reps: int = 2,
) -> pd.DataFrame:
    """Build a synthetic expression DataFrame with known ECHO parameters.

    Parameters
    ----------
    gamma_norm : float
        Amplitude change coefficient in *normalised* time [0, 1].  The fitter
        normalises t to [0, 1] internally, so this is what it will recover.
        To avoid numerical blow-up in the synthetic data, the helper converts
        gamma_norm to the equivalent per-hour² value before generating values:

            gamma_orig = gamma_norm / t_span²

        where t_span = (n_timepoints − 1) * tpoint_space (total hours).

    All other parameters control the sampling design and noise level.
    """
    rng = np.random.default_rng(seed)
    t_obs = np.arange(n_timepoints, dtype=float) * tpoint_space  # hours: 0, 2, ..., 22
    t_span = float(t_obs[-1] - t_obs[0])
    # Convert to per-hour² so that the fitter (which normalises t) recovers gamma_norm
    gamma_orig = gamma_norm / (t_span**2) if t_span > 0 else gamma_norm

    omega_hours = 2.0 * np.pi / 24.0  # 24-hour period
    phi = 0.0

    rows = {}
    for gene_i in range(n_genes):
        gene_id = f"gene_{gene_i}"
        row = {}
        for t in t_obs:
            clean = (
                A * np.exp(-gamma_orig * t**2) * np.cos(omega_hours * t + phi)
                + baseline
            )
            for rep in range(1, n_reps + 1):
                col = f"{zt_prefix}{int(t):02d}_{rep}"
                row[col] = clean + rng.normal(0, noise)
        rows[gene_id] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "#"
    return df


# ---------------------------------------------------------------------------
# Tests: _parse_timepoints helper
# ---------------------------------------------------------------------------


class TestParseTimepoints:
    def test_zt_columns(self):
        cols = ["ZT00_1", "ZT04_1", "ZT08_1"]
        result = _parse_timepoints(cols)
        np.testing.assert_array_equal(result, [0.0, 4.0, 8.0])

    def test_ct_columns(self):
        cols = ["CT02_1", "CT06_2"]
        result = _parse_timepoints(cols)
        np.testing.assert_array_equal(result, [2.0, 6.0])

    def test_three_digit_zt(self):
        cols = ["ZT100_1", "ZT120_2", "ZT140_1"]
        result = _parse_timepoints(cols)
        np.testing.assert_array_equal(result, [100.0, 120.0, 140.0])


# ---------------------------------------------------------------------------
# Tests: EchoFitter instantiation
# ---------------------------------------------------------------------------


class TestEchoFitterInit:
    def test_accepts_dataframe(self):
        df = _make_oscillation_df(gamma_norm=0.0)
        fitter = EchoFitter(df)
        assert fitter.data.shape == df.shape

    def test_accepts_tsv_path(self, tmp_path):
        df = _make_oscillation_df(gamma_norm=0.0)
        path = str(tmp_path / "expr.txt")
        df.to_csv(path, sep="\t")
        fitter = EchoFitter(path)
        assert fitter.data.shape == df.shape

    def test_accepts_parquet_path(self, tmp_path):
        df = _make_oscillation_df(gamma_norm=0.0)
        path = str(tmp_path / "expr.parquet")
        df.to_parquet(path)
        fitter = EchoFitter(path)
        assert fitter.data.shape == df.shape

    def test_reps_stored(self):
        df = _make_oscillation_df(gamma_norm=0.0)
        fitter = EchoFitter(df, reps=2)
        assert fitter.reps == 2

    def test_t_arr_normalised_to_unit_interval(self):
        df = _make_oscillation_df(gamma_norm=0.0, n_timepoints=12, tpoint_space=2)
        fitter = EchoFitter(df)
        assert float(fitter._t_arr[0]) == pytest.approx(0.0)
        assert float(fitter._t_arr[-1]) == pytest.approx(1.0)

    def test_mean_profiles_shape(self):
        df = _make_oscillation_df(gamma_norm=0.0, n_timepoints=12, n_reps=2)
        fitter = EchoFitter(df)
        # 12 unique timepoints → 12 columns in means DataFrame
        assert fitter._y_means.shape == (5, 12)


# ---------------------------------------------------------------------------
# Tests: fit() output structure
# ---------------------------------------------------------------------------


class TestEchoFitterFitOutput:
    @pytest.fixture(scope="class")
    def fit_result(self):
        df = _make_oscillation_df(gamma_norm=0.0)
        return EchoFitter(df).fit()

    def test_returns_dataframe(self, fit_result):
        assert isinstance(fit_result, pd.DataFrame)

    def test_all_output_columns_present(self, fit_result):
        for col in _ECHO_OUTPUT_COLS:
            assert col in fit_result.columns, f"Missing column: {col}"

    def test_index_matches_genes(self):
        df = _make_oscillation_df(gamma_norm=0.0, n_genes=3)
        result = EchoFitter(df).fit()
        assert set(result.index) == set(df.index)

    def test_echo_p_bh_in_unit_interval(self, fit_result):
        valid = fit_result["echo_p_bh"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_echo_converged_is_bool(self, fit_result):
        assert fit_result["echo_converged"].dtype == bool

    def test_amplitude_class_values(self, fit_result):
        classes = fit_result["echo_amplitude_class"].dropna().unique()
        valid = {"damped", "harmonic", "forced"}
        assert set(classes).issubset(valid)

    def test_index_name_preserved(self):
        df = _make_oscillation_df(gamma_norm=0.0)
        df.index.name = "#"
        result = EchoFitter(df).fit()
        assert result.index.name == "#"


# ---------------------------------------------------------------------------
# Tests: amplitude class recovery on synthetic data
# ---------------------------------------------------------------------------


class TestAmplitudeClassRecovery:
    def test_harmonic_gamma_near_zero(self):
        df = _make_oscillation_df(gamma_norm=0.0, noise=0.01, n_genes=10, seed=1)
        result = EchoFitter(df).fit()
        converged = result[result["echo_converged"]]
        assert len(converged) > 0
        assert (converged["echo_amplitude_class"] == "harmonic").all()

    def test_damped_gamma_positive(self):
        # gamma_norm=0.08 → amplitude at t̂=1 is exp(-0.08) ≈ 0.92 of A₀
        df = _make_oscillation_df(gamma_norm=0.08, noise=0.01, n_genes=10, seed=2)
        result = EchoFitter(df).fit()
        converged = result[result["echo_converged"]]
        assert len(converged) > 0
        assert (converged["echo_amplitude_class"] == "damped").all()

    def test_forced_gamma_negative(self):
        # gamma_norm=-0.08 → amplitude at t̂=1 is exp(0.08) ≈ 1.08 of A₀
        df = _make_oscillation_df(gamma_norm=-0.08, noise=0.01, n_genes=10, seed=3)
        result = EchoFitter(df).fit()
        converged = result[result["echo_converged"]]
        assert len(converged) > 0
        assert (converged["echo_amplitude_class"] == "forced").all()

    def test_gamma_within_bounds(self):
        """Fitted γ must stay within [−0.15, 0.15] (enforced by bounds)."""
        df = _make_oscillation_df(gamma_norm=0.0)
        result = EchoFitter(df).fit()
        valid_gamma = result["echo_gamma"].dropna()
        assert (valid_gamma >= -0.15).all() and (valid_gamma <= 0.15).all()


# ---------------------------------------------------------------------------
# Tests: multiprocessing
# ---------------------------------------------------------------------------


class TestMultiprocessing:
    def test_workers_gt1_same_results(self):
        df = _make_oscillation_df(gamma_norm=0.0, n_genes=6, noise=0.01, seed=42)
        r1 = EchoFitter(df).fit(workers=1)
        r2 = EchoFitter(df).fit(workers=2)
        assert set(r1.index) == set(r2.index)
        for col in ("echo_A", "echo_gamma", "echo_period"):
            pd.testing.assert_series_equal(
                r1[col].sort_index(),
                r2[col].sort_index(),
                check_names=False,
                rtol=1e-4,
            )


# ---------------------------------------------------------------------------
# Tests: column name edge cases
# ---------------------------------------------------------------------------


class TestColumnNameEdgeCases:
    def test_three_digit_zt_timepoints(self):
        """EchoFitter must handle 3+ digit ZT values (e.g. ZT100, ZT120)."""
        rng = np.random.default_rng(0)
        t_obs = [100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144]
        omega = 2.0 * np.pi / 24.0
        data = {}
        for t in t_obs:
            clean = 2.0 * np.cos(omega * t) + 5.0
            for rep in [1, 2]:
                col = f"ZT{t}_{rep}"
                data[col] = clean + rng.normal(0, 0.02)
        df = pd.DataFrame([data], index=["gene_big_zt"])
        df.index.name = "#"
        result = EchoFitter(df).fit()
        assert len(result) == 1
        assert "echo_amplitude_class" in result.columns

    def test_ct_prefix_columns(self):
        df = _make_oscillation_df(gamma_norm=0.0, zt_prefix="CT")
        result = EchoFitter(df).fit()
        assert len(result) == len(df)
