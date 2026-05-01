"""Expression classification integrating PIRS constitutiveness scores and BooteJTK rhythmicity."""

import os
import shutil
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

_REF_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "bootjtk", "ref_files"
    )
)

# (pirs_stable, rhythmic) -> label  — used when slope p-values are not available
_LABEL_MAP = {
    (True, True): "rhythmic",
    (True, False): "constitutive",
    (False, True): "noisy_rhythmic",
    (False, False): "variable",
}

# (pirs_stable, sloped, rhythmic) -> label  — used when slope p-values are available
_LABEL_MAP_SLOPE = {
    (True, False, True): "rhythmic",
    (True, False, False): "constitutive",
    (True, True, True): "rhythmic",
    (True, True, False): "linear",
    (False, False, True): "noisy_rhythmic",
    (False, False, False): "variable",
    (False, True, True): "noisy_rhythmic",
    (False, True, False): "linear",
}


class Classifier:
    """Classify expression profiles by combining PIRS and BooteJTK.

    Genes are assigned one of five labels based on two independent axes:

    - **constitutive**: stable expression, not rhythmic, not sloped
    - **rhythmic**: stable-to-moderate expression, strong circadian rhythm
    - **linear**: significant linear slope, not rhythmic
    - **variable**: high PIRS score (non-constitutive), not rhythmic, not sloped
    - **noisy_rhythmic**: high PIRS score with detectable rhythm signal

    ``linear`` is only emitted when slope p-values have been computed via
    :meth:`run_pirs` with ``slope_pvals=True``.

    Parameters
    ----------
    filename : str
        Path to the tab-separated expression file.  Must have a ``#`` index
        column and ``ZT``- or ``CT``-prefixed sample columns.
    anova : bool
        Whether PIRS should ANOVA-filter differentially expressed genes
        before computing scores.  Defaults to ``False`` so all genes receive
        a PIRS score and can be classified.
    reps : int
        Replicates per timepoint passed to BooteJTK (default 2).
    size : int
        Bootstrap iterations per gene in BooteJTK (default 50).
    workers : int
        Parallel worker processes for BooteJTK.  0 = all CPUs (default 1).
    """

    def __init__(
        self,
        source: str | Path | pd.DataFrame,
        *,
        anova: bool = False,
        reps: int = 2,
        size: int = 50,
        workers: int = 1,
    ) -> None:
        if isinstance(source, pd.DataFrame):
            self._source = source
            self.filename: str | None = None
        else:
            self._source = os.path.abspath(str(source))
            self.filename = self._source
        self.anova = anova
        self.reps = reps
        self.size = size
        self.workers = workers
        self.pirs_scores: pd.DataFrame | None = None
        self.rhythm_results: pd.DataFrame | None = None
        self.classifications: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Individual analysis steps
    # ------------------------------------------------------------------

    def run_pirs(
        self,
        pvals: bool = False,
        slope_pvals: bool = False,
        n_permutations: int = 1000,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Compute PIRS constitutiveness scores for every gene.

        Parameters
        ----------
        pvals : bool
            Whether to compute PIRS permutation p-values (left-tail temporal
            structure test).  Adds ``pval`` and ``pval_bh`` columns.
        slope_pvals : bool
            Whether to compute slope permutation p-values (right-tail slope
            test).  Adds ``slope_pval`` and ``slope_pval_bh`` columns.
        n_permutations : int
            Shuffles per gene passed to the permutation methods (default 1000).
        n_jobs : int
            Parallel workers passed to the permutation methods (default 1).

        Returns
        -------
        pd.DataFrame
            DataFrame with index = gene IDs, column ``score``, and optionally
            ``pval``, ``pval_bh``, ``slope_pval``, ``slope_pval_bh``.
        """
        from circ.pirs.rank import ranker

        r = ranker(self._source, anova=self.anova)
        r.get_tpoints()
        if self.anova:
            r.remove_anova()
        r.calculate_scores()
        if pvals:
            r.calculate_pvals(n_permutations=n_permutations, n_jobs=n_jobs)
        if slope_pvals:
            r.calculate_slope_pvals(n_permutations=n_permutations, n_jobs=n_jobs)
        self.pirs_scores = r.errors
        return self.pirs_scores

    def run_bootjtk(self, basic: bool = True) -> pd.DataFrame:
        """Run the BooteJTK + CalcP pipeline and return empirical results.

        The pipeline runs in an isolated temporary directory so no files are
        written alongside the original input.  The temp directory is removed
        when the run completes.

        Parameters
        ----------
        basic : bool
            When ``True`` (default), skip Limma/Vash variance shrinkage and
            run BooteJTK on the raw data directly.

        Returns
        -------
        pd.DataFrame
            BooteJTK results with empirical p-values added by CalcP.  Index is
            gene ID; columns include ``TauMean``, ``PeriodMean``, ``PhaseMean``,
            ``GammaBH``, etc.
        """
        from circ.bootjtk.pipeline import main as _pipeline_main

        workdir = tempfile.mkdtemp(prefix="circ_classify_")
        try:
            if isinstance(self._source, pd.DataFrame):
                fn_copy = os.path.join(workdir, "input.txt")
                self._source.to_csv(fn_copy, sep="\t")
            else:
                src = str(self._source)
                if src.endswith(".parquet"):
                    fn_copy = os.path.join(workdir, "input.txt")
                    pd.read_parquet(src).to_csv(fn_copy, sep="\t")
                else:
                    fn_copy = os.path.join(workdir, os.path.basename(src))
                    shutil.copy2(src, fn_copy)

            args = _make_pipeline_args(
                filename=fn_copy,
                ref_dir=_REF_DIR,
                reps=self.reps,
                size=self.size,
                workers=self.workers,
                basic=basic,
            )

            _pipeline_main(args)

            calcp_files = [f for f in os.listdir(workdir) if f.endswith("_GammaP.txt")]
            if not calcp_files:
                raise FileNotFoundError(
                    "BooteJTK/CalcP produced no *_GammaP.txt output in the "
                    f"working directory: {workdir}"
                )
            result_path = os.path.join(workdir, calcp_files[0])
            self.rhythm_results = pd.read_csv(result_path, sep="\t", index_col="ID")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        return self.rhythm_results

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        pirs_percentile: float = 50,
        slope_pval_threshold: float = 0.05,
        tau_threshold: float = 0.5,
        emp_p_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """Assign class labels using pre-computed PIRS and BooteJTK results.

        Call :meth:`run_pirs` and :meth:`run_bootjtk` first, or use
        :meth:`run_all` to do everything in one step.

        When slope p-values are present in ``pirs_scores`` (i.e. :meth:`run_pirs`
        was called with ``slope_pvals=True``), a five-label scheme is used that
        adds ``linear`` for genes with a significant slope that are not rhythmic.
        Otherwise the original four-label scheme is used.

        Parameters
        ----------
        pirs_percentile : float
            Genes whose PIRS score falls at or below this percentile of all
            scored genes are considered *stable* (default 50).
        slope_pval_threshold : float
            Maximum ``slope_pval`` for a gene to be called *sloped* (default
            0.05).  Only applied when ``slope_pval`` is present.
        tau_threshold : float
            Minimum ``TauMean`` required to call a gene rhythmic (default 0.5).
        emp_p_threshold : float
            Maximum ``GammaBH`` FDR-corrected p-value for a rhythmicity call
            (default 0.05).  Only applied when ``GammaBH`` is present.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by gene ID with columns:

            * ``pirs_score`` – raw PIRS score
            * ``pval``, ``pval_bh`` – PIRS permutation p-values, if computed
            * ``slope_pval``, ``slope_pval_bh`` – slope p-values, if computed
            * ``tau_mean`` – BooteJTK ``TauMean``
            * ``emp_p`` – FDR-corrected p-value (``GammaBH``), if available
            * ``period_mean`` – estimated period, if available
            * ``phase_mean`` – estimated phase, if available
            * ``tau_std`` – bootstrap standard deviation of TauMean
            * ``phase_std`` – circular SD of phase estimate (hours)
            * ``n_boots`` – number of BooteJTK bootstrap iterations
            * ``label`` – one of ``constitutive``, ``rhythmic``, ``linear``,
              ``variable``, ``noisy_rhythmic``
        """
        if self.pirs_scores is None:
            raise RuntimeError("Call run_pirs() before classify().")
        if self.rhythm_results is None:
            raise RuntimeError("Call run_bootjtk() before classify().")

        tau_col = "TauMean" if "TauMean" in self.rhythm_results.columns else "Tau"

        all_ids = self.pirs_scores.index.union(self.rhythm_results.index)
        result = pd.DataFrame(index=all_ids)
        result.index.name = self.pirs_scores.index.name

        result["pirs_score"] = self.pirs_scores["score"]

        for col in ("pval", "pval_bh", "slope_pval", "slope_pval_bh"):
            if col in self.pirs_scores.columns:
                result[col] = self.pirs_scores[col]

        result["tau_mean"] = self.rhythm_results[tau_col]

        for src_col, dst_col in [
            ("GammaBH", "emp_p"),
            ("PeriodMean", "period_mean"),
            ("PhaseMean", "phase_mean"),
            ("TauStdDev", "tau_std"),
            ("PhaseStdDev", "phase_std"),
            ("NumBoots", "n_boots"),
        ]:
            if src_col in self.rhythm_results.columns:
                result[dst_col] = self.rhythm_results[src_col]

        pirs_cutoff = np.percentile(result["pirs_score"].dropna(), pirs_percentile)
        stable = result["pirs_score"] <= pirs_cutoff

        rhythmic = result["tau_mean"] >= tau_threshold
        if "emp_p" in result.columns:
            rhythmic = rhythmic & (result["emp_p"] <= emp_p_threshold)

        if "slope_pval" in result.columns:
            sloped = result["slope_pval"] <= slope_pval_threshold
            result["label"] = [
                _LABEL_MAP_SLOPE.get((bool(s), bool(sl), bool(r)), "unclassified")
                for s, sl, r in zip(stable, sloped, rhythmic)
            ]
        else:
            result["label"] = [
                _LABEL_MAP.get((bool(s), bool(r)), "unclassified")
                for s, r in zip(stable, rhythmic)
            ]

        self.classifications = result
        return self.classifications

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def run_all(
        self,
        *,
        pvals: bool = False,
        slope_pvals: bool = False,
        n_permutations: int = 1000,
        n_jobs: int = 1,
        basic: bool = True,
        pirs_percentile: float = 50,
        slope_pval_threshold: float = 0.05,
        tau_threshold: float = 0.5,
        emp_p_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """Run PIRS, BooteJTK, and classify in a single call.

        Parameters mirror those of the individual methods.

        Returns
        -------
        pd.DataFrame
            Classification table as returned by :meth:`classify`.
        """
        self.run_pirs(
            pvals=pvals,
            slope_pvals=slope_pvals,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
        )
        self.run_bootjtk(basic=basic)
        return self.classify(
            pirs_percentile=pirs_percentile,
            slope_pval_threshold=slope_pval_threshold,
            tau_threshold=tau_threshold,
            emp_p_threshold=emp_p_threshold,
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _make_pipeline_args(
    *,
    filename: str,
    ref_dir: str,
    reps: int,
    size: int,
    workers: int,
    basic: bool,
) -> types.SimpleNamespace:
    """Build the argparse-compatible namespace that pipeline.main() expects."""
    return types.SimpleNamespace(
        filename=filename,
        means="DEFAULT",
        sds="DEFAULT",
        ns="DEFAULT",
        prefix="classify",
        waveform="cosine",
        period=os.path.join(ref_dir, "period24.txt"),
        phase=os.path.join(ref_dir, "phases_00-22_by2.txt"),
        width=os.path.join(ref_dir, "asymmetries_02-22_by2.txt"),
        output="DEFAULT",
        pickle="DEFAULT",
        id_list="DEFAULT",
        null_list="DEFAULT",
        size=size,
        reps=reps,
        workers=workers,
        write=False,
        basic=basic,
        limma=False,
        vash=False,
        noreps=False,
        rnaseq=False,
        harding=False,
        normal=False,
        jtk="DEFAULT",
        fit=False,
        null="DEFAULT",
    )
