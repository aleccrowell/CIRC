"""ECHO amplitude-aware circadian oscillator fitting.

Re-implements the core ECHO algorithm (De los Santos et al., Bioinformatics 2019)
in Python.  Each gene's mean expression across replicates per timepoint is fit
to the model

    x(t) = A · exp(−γt̂²) · cos(ωt̂ + φ) + y

where t̂ = (t − t_min) / t_span is the time normalised to [0, 1].  Using
normalised time makes the amplitude change coefficient γ dimensionless and
independent of the sampling interval, allowing the same classification
thresholds to be applied across experiments with different time spans.

The amplitude change coefficient γ classifies the oscillation as:

    damped   : γ >  0.03  (amplitude decreases over the data window)
    harmonic : |γ| ≤ 0.03 (approximately constant amplitude)
    forced   : γ < −0.03  (amplitude increases over the data window)

Reported period and phase are always converted back to the original time units
(e.g. hours).

Reference: De los Santos H et al. (2019). ECHO: an application for detection
and analysis of oscillators identifies metabolic regulation on genome-wide
circadian output. Bioinformatics, 36(3), 773–781.
"""

import multiprocessing

_mp_ctx = multiprocessing.get_context("forkserver")

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import kendalltau
from statsmodels.stats.multitest import multipletests

# γ classification thresholds (dimensionless; meaningful in normalised time)
_GAMMA_DAMPED_THRESHOLD = 0.03
_GAMMA_FORCED_THRESHOLD = -0.03
_GAMMA_MAX = 0.15

# Default period search range (hours) — covers the standard circadian window
_PERIOD_MIN = 18.0
_PERIOD_MAX = 36.0

_ECHO_OUTPUT_COLS = [
    "echo_A",
    "echo_gamma",
    "echo_period",
    "echo_phase",
    "echo_baseline",
    "echo_tau",
    "echo_p",
    "echo_p_bh",
    "echo_amplitude_class",
    "echo_converged",
]


def _echo_model(
    t: np.ndarray,
    A: float,
    gamma: float,
    omega: float,
    phi: float,
    baseline: float,
) -> np.ndarray:
    """ECHO oscillator model: A·exp(−γt²)·cos(ωt + φ) + y.

    t is expected to be in normalised [0, 1] coordinates.
    """
    return A * np.exp(-gamma * t**2) * np.cos(omega * t + phi) + baseline


def _classify_gamma(gamma: float) -> str:
    if gamma > _GAMMA_DAMPED_THRESHOLD:
        return "damped"
    if gamma < _GAMMA_FORCED_THRESHOLD:
        return "forced"
    return "harmonic"


def _parse_timepoints(columns) -> np.ndarray:
    """Extract numeric timepoint values from ZT/CT-prefixed column names."""
    tpoints = []
    for col in columns:
        c = str(col).replace("ZT", "").replace("CT", "")
        tpoints.append(int(c.split("_")[0]))
    return np.array(tpoints, dtype=float)


def _fit_gene(args: tuple) -> tuple:
    """Module-level worker: fit the ECHO model to one gene.

    Parameters
    ----------
    args : (gene_id, t_norm, y_arr, t_span)
        t_norm : np.ndarray — normalised timepoints in [0, 1]
        y_arr  : np.ndarray — mean expression per timepoint (may contain NaN)
        t_span : float — total time span in original units (e.g. hours)

    Returns
    -------
    tuple
        (gene_id, A, gamma, period_hours, phase_hours, baseline,
         tau, p_val, converged)
        All numeric values are NaN and converged=False on fitting failure.
    """
    gene_id, t_norm, y_arr, t_span = args
    _nan = (gene_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False)

    mask = ~np.isnan(y_arr)
    # Need at least 5 observations to constrain 5 free parameters
    if mask.sum() < 5:
        return _nan

    t_fit = t_norm[mask].astype(float)
    y_fit = y_arr[mask].astype(float)

    # Initial parameter guesses
    y_range = float(np.max(y_fit) - np.min(y_fit))
    A0 = y_range / 2.0 if y_range > 1e-10 else 1.0
    baseline0 = float(np.mean(y_fit))

    # omega in normalised-time units (period_hours → normalised period = period_hours/t_span)
    omega0 = 2.0 * np.pi * t_span / 24.0 if t_span > 0 else 2.0 * np.pi
    # Phase init: peak of expression → peak of cosine at that t_norm
    phi0 = float(-t_fit[int(np.argmax(y_fit))] * omega0)
    x0 = [A0, 0.0, omega0, phi0, baseline0]

    # Omega bounds derived from period bounds and the actual data span
    omega_lo = 2.0 * np.pi * t_span / _PERIOD_MAX if t_span > 0 else 2.0 * np.pi / 36.0
    omega_hi = 2.0 * np.pi * t_span / _PERIOD_MIN if t_span > 0 else 2.0 * np.pi / 18.0
    # Guard against inverted bounds when t_span < _PERIOD_MIN
    if omega_lo > omega_hi:
        omega_lo, omega_hi = omega_hi, omega_lo

    lb = [-np.inf, -_GAMMA_MAX, omega_lo, -2.0 * np.pi, -np.inf]
    ub = [np.inf, _GAMMA_MAX, omega_hi, 2.0 * np.pi, np.inf]

    try:
        popt, _ = curve_fit(
            _echo_model,
            t_fit,
            y_fit,
            p0=x0,
            bounds=(lb, ub),
            method="trf",
            maxfev=5000,
        )
        A_fit, gamma_fit, omega_fit, phi_fit, baseline_fit = popt

        y_fitted = _echo_model(t_fit, *popt)
        tau_stat, p_val = kendalltau(y_fit, y_fitted)

        # Convert normalised-time omega and phi back to original time units
        period_fit = float(2.0 * np.pi / omega_fit * t_span) if t_span > 0 else float(2.0 * np.pi / omega_fit)
        phase_hours = float((-phi_fit / omega_fit * t_span) % period_fit) if t_span > 0 else float((-phi_fit / omega_fit) % period_fit)
    except Exception:
        return _nan

    return (
        gene_id,
        float(A_fit),
        float(gamma_fit),
        period_fit,
        phase_hours,
        float(baseline_fit),
        float(tau_stat),
        float(p_val),
        True,
    )


class EchoFitter:
    """Fit the ECHO amplitude-aware circadian model to expression data.

    Each gene's mean expression across replicates per timepoint is fit to the
    normalised-time model:

        x(t̂) = A · exp(−γt̂²) · cos(ωt̂ + φ) + y

    where t̂ = (t − t_min) / t_span ∈ [0, 1].  The amplitude change coefficient
    γ classifies oscillations as damped (γ > 0.03), harmonic (|γ| ≤ 0.03), or
    forced (γ < −0.03).  Reported period and phase are in original time units.

    Parameters
    ----------
    source : str, Path, or pd.DataFrame
        Expression data with ZT/CT-prefixed sample columns (e.g. ``ZT02_1``).
        Loaded via ``circ.io.read_expression``.
    reps : int
        Nominal replicates per timepoint (informational; replicates are averaged
        automatically based on column names).
    """

    def __init__(self, source: Union[str, Path, pd.DataFrame], reps: int = 1) -> None:
        from circ.io import read_expression

        self.data = read_expression(source)
        self.reps = reps
        self._t_arr, self._t_span, self._y_means = self._build_mean_profiles()

    def _build_mean_profiles(self) -> tuple:
        """Average replicate columns; return normalised timepoints and mean profiles."""
        tpoints = _parse_timepoints(self.data.columns)
        unique_times = sorted(set(tpoints.tolist()))
        mean_cols: dict = {}
        for t in unique_times:
            mask = tpoints == t
            mean_cols[t] = self.data.iloc[:, mask].mean(axis=1)
        means_df = pd.DataFrame(mean_cols, index=self.data.index)

        t_raw = np.array(unique_times, dtype=float)
        t_min = t_raw[0]
        t_span = float(t_raw[-1] - t_min) if len(t_raw) > 1 else 1.0
        t_norm = (t_raw - t_min) / t_span if t_span > 0 else t_raw.copy()
        return t_norm, t_span, means_df

    def fit(self, workers: int = 1) -> pd.DataFrame:
        """Fit the ECHO model to every gene and return the results.

        Parameters
        ----------
        workers : int
            Parallel processes.  0 = all available CPUs.  Default 1.

        Returns
        -------
        pd.DataFrame
            Indexed by gene ID.  Columns:

            ``echo_A``              fitted amplitude
            ``echo_gamma``          amplitude change coefficient (normalised time)
            ``echo_period``         fitted period (original time units, e.g. hours)
            ``echo_phase``          fitted phase (original time units)
            ``echo_baseline``       fitted y-intercept
            ``echo_tau``            Kendall's τ goodness-of-fit
            ``echo_p``              p-value for τ
            ``echo_p_bh``           BH-corrected p-value
            ``echo_amplitude_class``  'damped', 'harmonic', or 'forced'
            ``echo_converged``      bool — whether optimisation converged
        """
        t_arr = self._t_arr
        t_span = self._t_span
        gene_args = [
            (gene_id, t_arr, self._y_means.loc[gene_id].values, t_span)
            for gene_id in self._y_means.index
        ]

        if workers == 1:
            raw = [_fit_gene(a) for a in gene_args]
        else:
            pool_size = workers if workers > 0 else None
            actual = pool_size or multiprocessing.cpu_count()
            chunksize = max(1, len(gene_args) // (actual * 4))
            with _mp_ctx.Pool(pool_size) as pool:
                raw = pool.map(_fit_gene, gene_args, chunksize=chunksize)

        records = {
            r[0]: {
                "echo_A": r[1],
                "echo_gamma": r[2],
                "echo_period": r[3],
                "echo_phase": r[4],
                "echo_baseline": r[5],
                "echo_tau": r[6],
                "echo_p": r[7],
                "echo_converged": r[8],
            }
            for r in raw
        }
        result = pd.DataFrame.from_dict(records, orient="index")
        result.index.name = self.data.index.name
        result["echo_converged"] = result["echo_converged"].astype(bool)

        # BH correction over all valid p-values
        valid_mask = result["echo_p"].notna()
        result["echo_p_bh"] = np.nan
        if valid_mask.any():
            _, pvals_bh, _, _ = multipletests(
                result.loc[valid_mask, "echo_p"].values, method="fdr_bh"
            )
            result.loc[valid_mask, "echo_p_bh"] = pvals_bh

        # Amplitude classification — only for converged fits with a valid γ
        result["echo_amplitude_class"] = None
        conv_mask = result["echo_converged"] & result["echo_gamma"].notna()
        if conv_mask.any():
            result.loc[conv_mask, "echo_amplitude_class"] = (
                result.loc[conv_mask, "echo_gamma"].apply(_classify_gamma)
            )

        return result[_ECHO_OUTPUT_COLS]
