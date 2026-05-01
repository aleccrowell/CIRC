"""
Python replacement for Limma_voom_core.R and Limma_voom_vash_core.R.

Implements variance-shrinkage preprocessing for circadian time-series data:
  - Sample-variance estimation from replicates (vooma equivalent)
  - Empirical Bayes variance shrinkage across (gene, timepoint) pairs
    following the Smyth (2004) method-of-moments estimator, matching
    limma's eBayes(robust=FALSE, trend=FALSE) behaviour
  - NA imputation before variance estimation (vash variant)

Public API
----------
run_vooma_ebayes(df_wide, period, rnaseq=False)
    Drop-in replacement for ``Limma_voom_core.R``.

run_vooma_vash(df_wide, period, rnaseq=False)
    Drop-in replacement for ``Limma_voom_vash_core.R``.

Both return a long-format DataFrame with columns:
    ID, Time, Mean, SD, SDpre, N
which is the format expected by ``limma_preprocess.write_limma_outputs``.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.special import polygamma


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _timepoint_groups(df, period):
    """Yield (time_mod_period, column_subset) for each unique ZT bucket."""
    tx = np.asarray(df.columns, dtype=float)
    for h in sorted(set(np.round(tx % period, 10))):
        mask = np.isclose(tx % period, h)
        yield float(h), df.iloc[:, mask]


def _vooma_stats(df_wide, period):
    """
    Compute per-(gene, timepoint) mean, sample SD, and N.

    Returns
    -------
    pd.DataFrame with columns: gene, time, mean, sd_pre, df, n
        sd_pre is NaN when fewer than 2 replicates are available.
        df is max(n - 1, 0).
    """
    rows = []
    for h, sub in _timepoint_groups(df_wide, period):
        vals = sub.to_numpy(dtype=float)
        n = np.sum(~np.isnan(vals), axis=1)
        means = np.nanmean(vals, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sds = np.nanstd(vals, axis=1, ddof=1)
        sds[n < 2] = np.nan
        dfs = np.where(n >= 2, n - 1, 0).astype(float)
        for i, gid in enumerate(df_wide.index):
            rows.append((gid, h, means[i], sds[i], dfs[i], int(n[i])))
    return pd.DataFrame(rows, columns=["gene", "time", "mean", "sd_pre", "df", "n"])


def _estimate_prior(sds, dfs):
    """
    Estimate empirical Bayes hyperparameters (d0, s0) via Smyth (2004)
    method-of-moments.

    Parameters
    ----------
    sds : array-like of float  – sample standard deviations
    dfs : array-like of float  – degrees of freedom per SD estimate

    Returns
    -------
    (d0, s0) : (float, float)
        Prior degrees of freedom and prior standard deviation.
        Returns (0, median(sds)) when there is insufficient data.
    """
    sds = np.asarray(sds, dtype=float)
    dfs = np.asarray(dfs, dtype=float)
    valid = np.isfinite(sds) & (sds > 0) & (dfs > 0)
    s, dg = sds[valid], dfs[valid]

    # Fallback: no shrinkage when data are too sparse
    fallback_s0 = float(np.nanmedian(sds[valid])) if valid.any() else 1.0
    if len(s) < 3:
        return 0.0, fallback_s0

    dg_fn = lambda x: polygamma(0, x)  # digamma
    tg_fn = lambda x: polygamma(1, x)  # trigamma
    ttg_fn = lambda x: polygamma(2, x)  # tetragamma

    def _solve_trigamma(x):
        """Solve trigamma(y) = x via Newton iteration (x > 0)."""
        if x <= 0:
            return 1.0
        y = 0.5 + 1.0 / x
        for _ in range(50):
            d = tg_fn(y) * (1.0 - tg_fn(y) / x) / ttg_fn(y)
            y += d
            if abs(d) < 1e-8 * abs(y):
                break
        return y

    G = len(s)
    z = 2.0 * np.log(s)
    e = z - dg_fn(dg / 2.0) + np.log(dg / 2.0)
    emean = float(np.nanmean(e))
    tri_rhs = float(np.nanmean((e - emean) ** 2 * G / (G - 1) - tg_fn(dg / 2.0)))
    if tri_rhs <= 0:
        return 0.0, float(np.exp(emean / 2.0))
    d0 = 2.0 * _solve_trigamma(tri_rhs)
    s0 = float(np.sqrt(np.exp(emean + dg_fn(d0 / 2.0) - np.log(d0 / 2.0))))
    return d0, s0


def _posterior_sd(sd_pre, dfs, d0, s0):
    """
    Compute limma posterior SD: sqrt((d0·s0² + df·sd²) / (d0 + df)).

    Features with no replicates (NaN sd_pre or df == 0) receive the prior s0.
    """
    sd_pre = np.asarray(sd_pre, dtype=float)
    dfs = np.asarray(dfs, dtype=float)
    has_data = np.isfinite(sd_pre) & (dfs > 0)
    numerator = np.where(has_data, d0 * s0**2 + dfs * sd_pre**2, d0 * s0**2)
    denominator = np.where(has_data, d0 + dfs, max(d0, 1e-10))
    return np.sqrt(numerator / denominator)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_vooma_ebayes(df_wide, period, rnaseq=False):
    """
    Python equivalent of ``Limma_voom_core.R``.

    For each (gene, timepoint) pair, computes the sample mean and SD from
    replicates (columns with the same value mod *period*), then shrinks the
    SDs toward a pooled prior using the Smyth (2004) empirical Bayes
    method-of-moments estimator.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Genes as rows; numeric-float columns representing timepoints.
        Columns sharing the same value mod *period* are treated as replicates.
    period : float
        Circadian period in hours (usually 24).
    rnaseq : bool
        Accepted for API compatibility; not used in this implementation.

    Returns
    -------
    pd.DataFrame with columns: ID, Time, Mean, SD, SDpre, N
    """
    long = _vooma_stats(df_wide, period)
    d0, s0 = _estimate_prior(long["sd_pre"].to_numpy(), long["df"].to_numpy())
    long["SD"] = _posterior_sd(long["sd_pre"].to_numpy(), long["df"].to_numpy(), d0, s0)
    return pd.DataFrame(
        {
            "ID": long["gene"].to_numpy(),
            "Time": long["time"].to_numpy(),
            "Mean": long["mean"].to_numpy(),
            "SD": long["SD"].to_numpy(),
            "SDpre": long["sd_pre"].to_numpy(),
            "N": long["n"].to_numpy(),
        }
    )


def _impute_na(df):
    """
    Impute missing values before variance estimation.

    Python equivalent of the R ``f_changeNA`` helper in
    ``Limma_voom_vash_core.R``:
      - Fully missing rows: replace with draws from N(grand_mean, grand_sd).
      - Partially missing rows: replace NAs with draws from
        N(row_mean, median_within_gene_sd).

    Parameters
    ----------
    df : pd.DataFrame  (genes × replicates, all numeric)

    Returns
    -------
    pd.DataFrame  – same shape, no NaN values.
    """
    vals = df.to_numpy(dtype=float).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row_sds = np.nanstd(vals, axis=1, ddof=1)
        col_means = np.nanmean(vals, axis=0)
    grand_mean = float(np.nanmean(col_means))
    grand_sd = float(np.nanstd(col_means, ddof=1)) if len(col_means) > 1 else 0.0
    finite_row_sds = row_sds[np.isfinite(row_sds)]
    med_sd = float(np.nanmedian(finite_row_sds)) if len(finite_row_sds) > 0 else 1.0

    rng = np.random.default_rng()
    for i in range(vals.shape[0]):
        row = vals[i]
        if np.all(np.isnan(row)):
            vals[i] = rng.normal(grand_mean, max(grand_sd, 1e-10), size=row.shape)
        elif np.any(np.isnan(row)):
            row_mean = float(np.nanmean(row))
            na_mask = np.isnan(row)
            vals[i, na_mask] = rng.normal(
                row_mean, max(med_sd, 1e-10), size=int(na_mask.sum())
            )

    return pd.DataFrame(vals, index=df.index, columns=df.columns)


def run_vooma_vash(df_wide, period, rnaseq=False):
    """
    Python equivalent of ``Limma_voom_vash_core.R``.

    Applies NA imputation per timepoint group before running the same
    vooma + eBayes variance shrinkage as :func:`run_vooma_ebayes`.
    This replaces the vashr ``vash()`` call with the Smyth (2004) eBayes
    estimator, which serves the same role of adaptive variance shrinkage.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Genes as rows; numeric-float columns representing timepoints.
    period : float
        Circadian period in hours (usually 24).
    rnaseq : bool
        Accepted for API compatibility; not used in this implementation.

    Returns
    -------
    pd.DataFrame with columns: ID, Time, Mean, SD, SDpre, N
    """
    tx = np.asarray(df_wide.columns, dtype=float)
    df_imp = df_wide.copy()
    for h in sorted(set(np.round(tx % period, 10))):
        mask = np.isclose(tx % period, h)
        if mask.sum() > 1:
            df_imp.iloc[:, mask] = _impute_na(df_wide.iloc[:, mask])
    return run_vooma_ebayes(df_imp, period, rnaseq=rnaseq)
