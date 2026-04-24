import multiprocessing

import pandas as pd
import numpy as np
from scipy.stats import f_oneway, t as t_dist
import statsmodels.stats.multitest as ssm
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Module-level helpers (must be at top level for multiprocessing picklability)
# ---------------------------------------------------------------------------

def _pirs_score(y, X_pinv, X_obs, X_fine, dof):
    """PIRS score for a single expression vector.

    score = max over fine grid of max(|PI_upper(t) - mean_expr|,
                                      |PI_lower(t) - mean_expr|) / |mean_expr|

    The 95% prediction interval bounds capture both trend (via the fitted
    slope) and residual noise (via the residual SE).  A flat, low-noise gene
    has bounds close to mean expression everywhere → score near 0.  A trending
    or noisy gene has bounds that deviate far from mean expression → large score.

    Parameters
    ----------
    y      : ndarray (n_obs,)
    X_pinv : ndarray (2, n_obs)  — pseudoinverse of design matrix
    X_obs  : ndarray (n_obs, 2)  — design matrix at observed timepoints
    X_fine : ndarray (n_fine, 2) — design matrix at fine grid
    dof    : int  — number of unique timepoints (used as df for residual SE)
    """
    n = len(y)
    beta = X_pinv @ y
    y_hat_obs  = X_obs  @ beta
    y_hat_fine = X_fine @ beta

    mean_expr = np.mean(y)
    denom = abs(mean_expr) if abs(mean_expr) > 1e-10 else 1.0

    rss = np.sum((y - y_hat_obs) ** 2)
    df_resid = max(dof - 2, 1)
    s = np.sqrt(max(rss, 1e-28) / df_resid)
    t_crit = t_dist.ppf(0.975, df_resid)

    x_obs  = X_obs[:, 1]
    x_fine = X_fine[:, 1]
    x_mean = np.mean(x_obs)
    Sxx = np.sum((x_obs - x_mean) ** 2)
    lev = (1.0 + 1.0 / n + (x_fine - x_mean) ** 2 / Sxx
           if Sxx > 1e-14 else np.full(len(x_fine), 1.0 + 1.0 / n))

    pi_half = t_crit * s * np.sqrt(lev)
    upper = y_hat_fine + pi_half
    lower = y_hat_fine - pi_half

    max_dev = np.max(np.maximum(np.abs(upper - mean_expr), np.abs(lower - mean_expr)))
    return float(max_dev / denom)


def _permutation_worker(args):
    """Multiprocessing worker: permute y n_permutations times, return p-value.

    Uses a left-tail test: counts how many permuted PIRS scores are <= the
    observed score.  When a gene has real temporal structure the linear fit is
    good, residuals are small, and the observed PIRS score sits in the lower
    tail of the null distribution (shuffling breaks the fit, inflating s and
    widening the PI bounds).  Small p therefore means significantly
    non-constitutive.
    """
    gene_id, y, X_pinv, X_obs, X_fine, dof, obs_score, n_perm, seed = args
    rng = np.random.default_rng(seed)
    y_perm = y.copy()
    count_le = 0
    for _ in range(n_perm):
        rng.shuffle(y_perm)
        if _pirs_score(y_perm, X_pinv, X_obs, X_fine, dof) <= obs_score:
            count_le += 1
    # Include the observed value as one realisation of the permutation distribution
    return gene_id, (count_le + 1) / (n_perm + 1)


def _slope_score(y, X_pinv, X_fine):
    """Slope-only score: max regression trend deviation from mean, normalized by mean.

    score = max(|ŷ_fine(t) − mean_expr|) / |mean_expr|

    No noise term — only the fitted trend contributes.  After permutation the
    estimated slope collapses toward zero, making this suitable for a right-tail
    test that detects statistically significant linear slope.
    """
    beta = X_pinv @ y
    y_hat_fine = X_fine @ beta
    mean_expr = np.mean(y)
    denom = abs(mean_expr) if abs(mean_expr) > 1e-10 else 1.0
    return float(np.max(np.abs(y_hat_fine - mean_expr)) / denom)


def _slope_permutation_worker(args):
    """Multiprocessing worker for the slope permutation test.

    Right-tail test: counts how many permuted slope scores are >= the observed
    score.  Shuffling destroys linear trend structure, so the slope score
    collapses toward zero for genuinely sloped genes; the observed score sits
    in the upper tail.  Small p means significantly non-flat.
    """
    gene_id, y, X_pinv, X_fine, obs_score, n_perm, seed = args
    rng = np.random.default_rng(seed)
    y_perm = y.copy()
    count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(y_perm)
        if _slope_score(y_perm, X_pinv, X_fine) >= obs_score:
            count_ge += 1
    return gene_id, (count_ge + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# ranker
# ---------------------------------------------------------------------------

class ranker:
    """Ranks and sorts expression profiles from most to least constitutive.

    Fits a linear regression to each time series, computes prediction
    intervals, and scores each profile by how much the regression trend
    deviates from the mean expression normalized by the interval half-width.
    A score near 0 means the gene is constitutive; a large score means the
    gene has a real, statistically meaningful trend away from its mean.

    Parameters
    ----------
    filename : str
        Path to the input dataset.
    anova : bool
        Whether to ANOVA-filter differentially expressed profiles before
        scoring.

    Attributes
    ----------
    data   : DataFrame  — expression data
    anova  : bool
    errors : DataFrame  — scores (and optionally p-values) after
                          calculate_scores() / calculate_pvals()
    """

    def __init__(self, filename, anova=True):
        self.data = pd.read_csv(filename, sep='\t', header=0, index_col=0)
        self.data = self.data[(self.data.T != 0).any()]
        self.anova = anova
        self.errors = None

    def get_tpoints(self):
        """Extract numeric timepoints from column headers (ZT/CT prefix)."""
        tpoints = [i.replace('ZT', '').replace('CT', '') for i in self.data.columns.values]
        tpoints = [int(i.split('_')[0]) for i in tpoints]
        self.tpoints = np.asarray(tpoints)

    def remove_anova(self, alpha=0.05):
        """Remove profiles with significant differential expression (one-way ANOVA).

        Parameters
        ----------
        alpha : float
            P-value threshold; profiles below this are removed (default 0.05).
        """
        to_remove = []
        for index, row in self.data.iterrows():
            vals = []
            for i in list(set(self.tpoints)):
                vals.append([row.values[j] for j in range(len(row)) if self.tpoints[j] == i])
            f_val, p_val = f_oneway(*vals)
            if p_val < alpha:
                to_remove.append(index)
        self.data = self.data[~self.data.index.isin(to_remove)]

    def calculate_scores(self):
        """Compute PIRS scores for every expression profile.

        score = max over fine grid of max(|PI_upper(t) - mean_expr|,
                                          |PI_lower(t) - mean_expr|) / |mean_expr|

        The 95% prediction interval bounds capture both trend and residual
        noise.  Flat, low-noise genes score near 0; trending or noisy genes
        score higher.

        Returns
        -------
        DataFrame
            Index = gene IDs, column ``score``, sorted ascending.
        """
        tpoints = self.tpoints
        dof     = len(np.unique(tpoints))
        x_fine  = np.arange(min(tpoints), max(tpoints), 0.1)

        # Pre-compute design matrices — shared across all genes
        X_obs    = np.column_stack([np.ones(len(tpoints)), tpoints])
        X_pinv   = np.linalg.pinv(X_obs)
        X_fine_m = np.column_stack([np.ones(len(x_fine)), x_fine])

        es = {}
        for index in tqdm(range(len(self.data))):
            y = np.array(self.data.iloc[index], dtype=float)
            es[index] = _pirs_score(y, X_pinv, X_obs, X_fine_m, dof)

        self.errors = pd.DataFrame.from_dict(es, orient='index')
        self.errors.columns = ['score']
        self.errors.index = self.data.index
        self.errors.sort_values('score', inplace=True)
        return self.errors

    def calculate_pvals(self, n_permutations=1000, n_jobs=1):
        """Compute permutation p-values for PIRS scores.

        For each gene, shuffles the expression values ``n_permutations``
        times (breaking timepoint structure while preserving the marginal
        distribution) and re-computes the PIRS score.  A left-tail test is
        used: when temporal structure is present the linear fit is good,
        residuals are small, and the observed score sits below most of the
        null distribution.  Small p therefore means the gene is significantly
        non-constitutive.

        Benjamini–Hochberg FDR correction is applied across all genes.

        Requires :meth:`calculate_scores` to have been called first.

        Parameters
        ----------
        n_permutations : int
            Shuffles per gene (default 1000).
        n_jobs : int
            Parallel worker processes.  0 = all CPUs (default 1).

        Returns
        -------
        DataFrame
            ``self.errors`` with added columns ``pval`` and ``pval_bh``.
        """
        if self.errors is None:
            raise RuntimeError("Call calculate_scores() before calculate_pvals().")

        tpoints = self.tpoints
        dof     = len(np.unique(tpoints))
        x_fine  = np.arange(min(tpoints), max(tpoints), 0.1)

        X_obs    = np.column_stack([np.ones(len(tpoints)), tpoints])
        X_pinv   = np.linalg.pinv(X_obs)
        X_fine_m = np.column_stack([np.ones(len(x_fine)), x_fine])

        # Build one arg-tuple per gene; seed is fixed per gene for reproducibility
        gene_args = [
            (gene_id,
             np.array(self.data.loc[gene_id], dtype=float),
             X_pinv, X_obs, X_fine_m, dof,
             float(self.errors.loc[gene_id, 'score']),
             n_permutations,
             i)
            for i, gene_id in enumerate(self.errors.index)
        ]

        if n_jobs == 1:
            results = [_permutation_worker(a) for a in tqdm(gene_args)]
        else:
            pool_size = n_jobs if n_jobs > 0 else None
            actual    = pool_size or multiprocessing.cpu_count()
            chunksize = max(1, len(gene_args) // (actual * 4))
            with multiprocessing.Pool(pool_size) as pool:
                results = list(tqdm(
                    pool.imap(_permutation_worker, gene_args, chunksize=chunksize),
                    total=len(gene_args),
                ))

        pvals = pd.Series(
            {gid: pval for gid, pval in results},
            name='pval',
        ).reindex(self.errors.index)

        _, pvals_bh, _, _ = ssm.multipletests(pvals.values, method='fdr_bh')

        self.errors['pval']    = pvals.values
        self.errors['pval_bh'] = pvals_bh
        return self.errors

    def calculate_slope_pvals(self, n_permutations=1000, n_jobs=1):
        """Compute permutation p-values for the slope component of expression.

        For each gene, shuffles expression values ``n_permutations`` times and
        re-computes the slope score (max fitted-trend deviation from mean,
        normalized by mean expression).  Shuffling destroys linear trend
        structure, so the slope score collapses toward zero; the observed score
        sits in the upper tail for genuinely sloped genes.

        A right-tail test is used: small p means the gene has a statistically
        significant linear slope.  Benjamini–Hochberg FDR correction is applied.

        Requires :meth:`calculate_scores` to have been called first.

        Parameters
        ----------
        n_permutations : int
            Shuffles per gene (default 1000).
        n_jobs : int
            Parallel worker processes.  0 = all CPUs (default 1).

        Returns
        -------
        DataFrame
            ``self.errors`` with added columns ``slope_pval`` and
            ``slope_pval_bh``.
        """
        if self.errors is None:
            raise RuntimeError("Call calculate_scores() before calculate_slope_pvals().")

        tpoints = self.tpoints
        x_fine  = np.arange(min(tpoints), max(tpoints), 0.1)

        X_obs    = np.column_stack([np.ones(len(tpoints)), tpoints])
        X_pinv   = np.linalg.pinv(X_obs)
        X_fine_m = np.column_stack([np.ones(len(x_fine)), x_fine])

        gene_args = [
            (gene_id,
             np.array(self.data.loc[gene_id], dtype=float),
             X_pinv, X_fine_m,
             _slope_score(np.array(self.data.loc[gene_id], dtype=float), X_pinv, X_fine_m),
             n_permutations,
             i)
            for i, gene_id in enumerate(self.errors.index)
        ]

        if n_jobs == 1:
            results = [_slope_permutation_worker(a) for a in tqdm(gene_args)]
        else:
            pool_size = n_jobs if n_jobs > 0 else None
            actual    = pool_size or multiprocessing.cpu_count()
            chunksize = max(1, len(gene_args) // (actual * 4))
            with multiprocessing.Pool(pool_size) as pool:
                results = list(tqdm(
                    pool.imap(_slope_permutation_worker, gene_args, chunksize=chunksize),
                    total=len(gene_args),
                ))

        pvals = pd.Series(
            {gid: pval for gid, pval in results},
            name='slope_pval',
        ).reindex(self.errors.index)

        _, pvals_bh, _, _ = ssm.multipletests(pvals.values, method='fdr_bh')

        self.errors['slope_pval']    = pvals.values
        self.errors['slope_pval_bh'] = pvals_bh
        return self.errors

    def pirs_sort(self, outname=False, pvals=False, slope_pvals=False,
                  n_permutations=1000, n_jobs=1):
        """Run the full PIRS pipeline and return data sorted by score.

        Parameters
        ----------
        outname : str
            If provided, write scores (and p-values if computed) to this
            tab-separated file.
        pvals : bool
            Whether to compute PIRS permutation p-values (default False).
        slope_pvals : bool
            Whether to compute slope permutation p-values (default False).
        n_permutations : int
            Passed to :meth:`calculate_pvals` / :meth:`calculate_slope_pvals`
            (default 1000).
        n_jobs : int
            Passed to :meth:`calculate_pvals` / :meth:`calculate_slope_pvals`
            (default 1).

        Returns
        -------
        DataFrame
            Input expression data sorted by PIRS score (most constitutive
            first).
        """
        self.get_tpoints()
        if self.anova:
            self.remove_anova()
        self.calculate_scores()
        if pvals:
            self.calculate_pvals(n_permutations=n_permutations, n_jobs=n_jobs)
        if slope_pvals:
            self.calculate_slope_pvals(n_permutations=n_permutations, n_jobs=n_jobs)
        sorted_data = self.data.loc[self.errors.index.values]
        if outname:
            self.errors.to_csv(outname, sep='\t')
        return sorted_data


# ---------------------------------------------------------------------------
# rsd_ranker  (unchanged — benchmarking baseline)
# ---------------------------------------------------------------------------

class rsd_ranker:
    """Ranks profiles by Relative Standard Deviation (benchmarking baseline).

    Parameters
    ----------
    filename : str
        Path to the input dataset.

    Attributes
    ----------
    data : DataFrame
    """

    def __init__(self, filename):
        self.data = pd.read_csv(filename, sep='\t', header=0, index_col=0)
        self.data = self.data[(self.data.T != 0).any()]

    def calculate_scores(self):
        """Compute RSD scores (sorted ascending)."""
        rsd = (1 + (1 / (4 * len(self.data)))) * np.std(self.data.values, axis=1) / np.abs(np.mean(self.data.values, axis=1))
        self.rsd = pd.DataFrame(rsd, index=self.data.index)
        self.rsd.columns = ['score']
        self.rsd.sort_values('score', inplace=True)
        return self.rsd

    def rsd_sort(self, outname=False):
        """Run pipeline and return data sorted by RSD.

        Parameters
        ----------
        outname : str
            If provided, write scores to this file.

        Returns
        -------
        DataFrame
        """
        self.calculate_scores()
        sorted_data = self.data.loc[self.rsd.index.values]
        if outname:
            self.rsd.to_csv(outname, sep='\t')
        return sorted_data
