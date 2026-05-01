"""Statistical comparison of classification results between two experimental conditions.

The main entry point is :func:`compare_conditions`, which takes the result
DataFrames from two :class:`~circ.expression_classification.classify.Classifier`
runs and returns a per-gene summary of effect sizes and, when BooteJTK
uncertainty columns (``tau_std``, ``phase_std``, ``n_boots``) are present,
FDR-corrected significance tests.

Cross-omic comparisons (proteomics vs. gene expression)
-------------------------------------------------------
Proteomics data is classified at peptide level and carries a
``(Peptide, Protein)`` MultiIndex.  Before passing such results to
:func:`compare_conditions`, collapse them to protein level with
:func:`aggregate_to_protein`.  The Protein identifiers must then overlap
with the gene IDs in the expression result.

Statistical methods
-------------------
**Rhythmicity change (TauMean)**
    Welch's t-test on the bootstrap tau distributions.  BooteJTK reports
    ``tau_std`` (SD of best-match tau values across ``n_boots`` bootstrap
    resamples) and ``n_boots``, giving enough information to construct a
    two-sample t-statistic with Welch–Satterthwaite degrees of freedom.

**Phase shift**
    A z-test on the signed circular phase difference B − A (wrapped to
    ±12 h).  Standard errors are approximated as ``phase_std / √n_boots``
    for each condition.  Only genes called rhythmic in *both* conditions
    are tested; phase estimates for non-rhythmic genes are unreliable.

Both tests apply Benjamini–Hochberg FDR correction across all tested genes.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_to_protein(result):
    """Aggregate peptide-level classifier results to the protein level.

    When proteomics data is classified at peptide level the result carries a
    ``(Peptide, Protein)`` MultiIndex.  This function collapses it to a
    single-level index on ``Protein`` so the result can be compared with gene
    expression results via :func:`compare_conditions`.

    Aggregation rules:

    * **numeric columns** — mean across all peptides for the same protein
    * **phase_mean** — circular mean (period 24 h) to handle wrap-around
    * **label** — majority vote; ties broken by the first label alphabetically

    Parameters
    ----------
    result : pd.DataFrame
        Classifier output with a ``(Peptide, Protein)`` MultiIndex, as
        produced when the expression data uses a proteomics-style multi-index.
        If *result* already has a flat index it is returned unchanged.

    Returns
    -------
    pd.DataFrame
        Same columns as *result*, indexed by ``Protein``.
    """
    if not isinstance(result.index, pd.MultiIndex):
        return result.copy()

    protein = result.index.get_level_values("Protein")
    df = result.copy()
    df.index = pd.Index(protein, name="Protein")

    numeric_cols = set(df.select_dtypes(include="number").columns)
    agg = {}
    for col in df.columns:
        if col == "phase_mean":
            continue
        elif col == "label":
            agg[col] = _majority_label
        elif col in numeric_cols:
            agg[col] = "mean"

    out = (
        df.groupby(df.index, sort=False).agg(agg)
        if agg
        else pd.DataFrame(index=df.index.unique())
    )

    if "phase_mean" in df.columns:
        angles = 2 * np.pi * df["phase_mean"].values / 24.0
        sin_s = pd.Series(np.sin(angles), index=df.index)
        cos_s = pd.Series(np.cos(angles), index=df.index)
        sin_mean = sin_s.groupby(df.index, sort=False).mean()
        cos_mean = cos_s.groupby(df.index, sort=False).mean()
        out["phase_mean"] = (np.arctan2(sin_mean, cos_mean) * 24 / (2 * np.pi)) % 24

    return out.reindex(columns=[c for c in result.columns if c in out.columns])


def compare_conditions(
    result_A,
    result_B,
    emp_p_threshold=0.05,
    tau_threshold=0.5,
):
    """Compare classification results for shared genes between two conditions.

    For each gene present in both result DataFrames, computes effect sizes
    (``delta_tau``, ``delta_pirs``, ``delta_phase``) and, when BooteJTK
    uncertainty columns are present (``tau_std``, ``phase_std``,
    ``n_boots``), significance tests with Benjamini–Hochberg correction.

    Parameters
    ----------
    result_A, result_B : pd.DataFrame
        Output of ``Classifier.classify()`` for each condition.  Both must
        share at least some gene IDs in their index.
    emp_p_threshold : float
        GammaBH threshold for calling a gene rhythmic (default 0.05).
    tau_threshold : float
        TauMean threshold for calling a gene rhythmic (default 0.5).

    Returns
    -------
    pd.DataFrame
        One row per shared gene.  Always-present columns:

        * ``label_A``, ``label_B`` — expression label in each condition
        * ``rhythmicity_status`` — ``"gained"``, ``"lost"``,
          ``"maintained_rhythmic"``, or ``"maintained_nonrhythmic"``
        * ``tau_mean_A``, ``tau_mean_B``, ``delta_tau``
        * ``pirs_score_A``, ``pirs_score_B``, ``delta_pirs``
        * ``emp_p_A``, ``emp_p_B`` — per-condition GammaBH (if in input)
        * ``phase_A``, ``phase_B``, ``delta_phase`` — circular phase
          difference in hours (±12 h); ``NaN`` for genes not rhythmic
          in both conditions (if phase columns present)

        Significance columns (added when ``tau_std`` and ``n_boots`` are
        present in both DataFrames):

        * ``tau_pval``, ``tau_padj`` — Welch t-test on TauMean difference
        * ``phase_pval``, ``phase_padj`` — z-test on phase shift
          (only for genes rhythmic in both conditions)
    """
    for _name, _res in (("result_A", result_A), ("result_B", result_B)):
        if isinstance(_res.index, pd.MultiIndex):
            raise ValueError(
                f"{_name} has a MultiIndex (peptide-level proteomics data). "
                "Call aggregate_to_protein(result) first to collapse to protein "
                "level before comparing."
            )

    shared = result_A.index.intersection(result_B.index)
    if len(shared) == 0:
        raise ValueError(
            "result_A and result_B share no gene IDs. "
            "Check that both DataFrames are indexed by the same identifiers. "
            "For proteomics vs. gene expression comparisons, call "
            "aggregate_to_protein() first and ensure Protein IDs match gene IDs."
        )

    A = result_A.loc[shared]
    B = result_B.loc[shared]

    out = pd.DataFrame(index=shared)
    out.index.name = A.index.name
    out["label_A"] = A["label"]
    out["label_B"] = B["label"]

    rhythmic_A = _is_rhythmic(A, emp_p_threshold, tau_threshold)
    rhythmic_B = _is_rhythmic(B, emp_p_threshold, tau_threshold)
    out["rhythmicity_status"] = _rhythmicity_status(rhythmic_A, rhythmic_B)

    # TauMean effect size
    out["tau_mean_A"] = A["tau_mean"]
    out["tau_mean_B"] = B["tau_mean"]
    out["delta_tau"] = B["tau_mean"].values - A["tau_mean"].values

    # PIRS effect size
    out["pirs_score_A"] = A["pirs_score"]
    out["pirs_score_B"] = B["pirs_score"]
    out["delta_pirs"] = B["pirs_score"].values - A["pirs_score"].values

    # Empirical p-values (pass through for reference)
    for suffix, df in [("_A", A), ("_B", B)]:
        if "emp_p" in df.columns:
            out[f"emp_p{suffix}"] = df["emp_p"]

    # Phase effect size (circular)
    if "phase_mean" in A.columns and "phase_mean" in B.columns:
        out["phase_A"] = A["phase_mean"]
        out["phase_B"] = B["phase_mean"]
        raw_diff = _circular_diff(A["phase_mean"].values, B["phase_mean"].values)
        out["delta_phase"] = raw_diff
        # Phase difference is only meaningful when rhythmic in both conditions
        out.loc[~(rhythmic_A & rhythmic_B), "delta_phase"] = np.nan

    # Significance tests (requires uncertainty columns from BooteJTK)
    has_uncertainty = all(c in A.columns for c in ("tau_std", "n_boots")) and all(
        c in B.columns for c in ("tau_std", "n_boots")
    )
    if has_uncertainty:
        _tau_test(out, A, B)
        if (
            "delta_phase" in out.columns
            and "phase_std" in A.columns
            and "phase_std" in B.columns
        ):
            _phase_test(out, A, B)

    return out


def label_change_table(comparison):
    """Pivot table of label transitions between conditions (A → B).

    Parameters
    ----------
    comparison : pd.DataFrame
        Output of :func:`compare_conditions`.

    Returns
    -------
    pd.DataFrame
        Rows = label in condition A, columns = label in condition B,
        values = gene count.  Labels are ordered by ``_LABEL_ORDER``.
    """
    from circ.visualization.classify import _LABEL_ORDER

    all_labels = list(
        dict.fromkeys(
            [
                l
                for l in _LABEL_ORDER
                if l in comparison["label_A"].values
                or l in comparison["label_B"].values
            ]
        )
    )
    ct = pd.crosstab(comparison["label_A"], comparison["label_B"])
    return ct.reindex(index=all_labels, columns=all_labels, fill_value=0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _majority_label(series):
    """Return the most common label; ties broken alphabetically."""
    return series.mode().iloc[0]


def _circular_diff(a, b, period=24.0):
    """Signed circular difference b − a, wrapped to (−period/2, +period/2]."""
    diff = (np.asarray(b, dtype=float) - np.asarray(a, dtype=float)) % period
    diff = np.where(diff > period / 2, diff - period, diff)
    return diff


def _bh_correct(pvals):
    """Benjamini–Hochberg FDR correction; NaNs are preserved."""
    pvals = np.asarray(pvals, dtype=float)
    result = np.full(len(pvals), np.nan)
    valid = ~np.isnan(pvals)
    if not valid.any():
        return result
    p = pvals[valid]
    m = len(p)
    order = np.argsort(p)
    rank = np.empty(m, dtype=float)
    rank[order] = np.arange(1, m + 1)
    adj = p * m / rank
    # Enforce monotone decrease from right so smaller ranks stay ≤ larger
    adj_sorted = adj[order]
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj[order] = np.minimum(adj_sorted, 1.0)
    result[valid] = adj
    return result


def _is_rhythmic(result, emp_p_threshold, tau_threshold):
    """Boolean Series: True if the gene passes rhythmicity thresholds."""
    has_emp_p = "emp_p" in result.columns and result["emp_p"].notna().any()
    has_tau = "tau_mean" in result.columns and result["tau_mean"].notna().any()
    if has_emp_p and has_tau:
        return (result["emp_p"] < emp_p_threshold) & (
            result["tau_mean"] >= tau_threshold
        )
    elif has_tau:
        return result["tau_mean"] >= tau_threshold
    return result["label"].isin(["rhythmic", "noisy_rhythmic"])


def _rhythmicity_status(rhythmic_A, rhythmic_B):
    status = pd.Series("maintained_nonrhythmic", index=rhythmic_A.index, dtype=object)
    status[rhythmic_A & rhythmic_B] = "maintained_rhythmic"
    status[~rhythmic_A & rhythmic_B] = "gained"
    status[rhythmic_A & ~rhythmic_B] = "lost"
    return status


def _tau_test(out, A, B):
    """Vectorised Welch's t-test on bootstrap TauMean distributions."""
    ta = A["tau_mean"].values.astype(float)
    sa = A["tau_std"].values.astype(float)
    na = A["n_boots"].values.astype(float)
    tb = B["tau_mean"].values.astype(float)
    sb = B["tau_std"].values.astype(float)
    nb = B["n_boots"].values.astype(float)

    se_a_sq = np.where(na > 1, sa**2 / na, np.nan)
    se_b_sq = np.where(nb > 1, sb**2 / nb, np.nan)
    se = np.sqrt(se_a_sq + se_b_sq)
    t = (tb - ta) / se

    # Welch–Satterthwaite degrees of freedom
    df_ws = (se_a_sq + se_b_sq) ** 2 / (
        se_a_sq**2 / np.maximum(na - 1, 1) + se_b_sq**2 / np.maximum(nb - 1, 1)
    )
    pvals = 2 * stats.t.sf(np.abs(t), df_ws)
    pvals[~np.isfinite(t)] = np.nan

    out["tau_pval"] = pvals
    out["tau_padj"] = _bh_correct(pvals)


def _phase_test(out, A, B):
    """Vectorised z-test on circular phase shift (genes rhythmic in both)."""
    delta = out["delta_phase"].values.astype(float)  # NaN where not both rhythmic
    sa = A["phase_std"].values.astype(float)
    na = A["n_boots"].values.astype(float)
    sb = B["phase_std"].values.astype(float)
    nb = B["n_boots"].values.astype(float)

    se = np.sqrt(sa**2 / np.maximum(na, 1) + sb**2 / np.maximum(nb, 1))
    z = delta / se  # NaN propagates from delta
    pvals = 2 * stats.norm.sf(np.abs(z))
    pvals[~np.isfinite(z)] = np.nan

    out["phase_pval"] = pvals
    out["phase_padj"] = _bh_correct(pvals)
