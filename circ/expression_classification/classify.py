"""Expression classification integrating PIRS constitutiveness scores and BooteJTK rhythmicity."""
import os
import shutil
import tempfile
import types

import numpy as np
import pandas as pd

_REF_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bootjtk', 'ref_files')
)

# (pirs_stable, rhythmic) -> label
_LABEL_MAP = {
    (True,  True):  'rhythmic',
    (True,  False): 'constitutive',
    (False, True):  'noisy_rhythmic',
    (False, False): 'variable',
}


class Classifier:
    """Classify expression profiles by combining PIRS and BooteJTK.

    Genes are assigned one of four labels based on two independent axes:

    - **constitutive**: stable expression (low PIRS score), not rhythmic
    - **rhythmic**: stable-to-moderate expression, strong circadian rhythm
    - **variable**: high PIRS score (non-constitutive), not rhythmic
    - **noisy_rhythmic**: high PIRS score with detectable rhythm signal

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

    def __init__(self, filename, *, anova=False, reps=2, size=50, workers=1):
        self.filename = os.path.abspath(filename)
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

    def run_pirs(self, alpha: float = 0.5) -> pd.DataFrame:
        """Compute PIRS constitutiveness scores for every gene.

        Parameters
        ----------
        alpha : float
            Significance level for prediction interval calculation (default 0.5).

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame with index = gene IDs and column ``score``.
        """
        from circ.pirs.rank import ranker

        r = ranker(self.filename, anova=self.anova)
        r.get_tpoints()
        if self.anova:
            r.remove_anova()
        self.pirs_scores = r.calculate_scores(alpha=alpha)
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

        workdir = tempfile.mkdtemp(prefix='circ_classify_')
        try:
            fn_copy = os.path.join(workdir, os.path.basename(self.filename))
            shutil.copy2(self.filename, fn_copy)

            args = _make_pipeline_args(
                filename=fn_copy,
                ref_dir=_REF_DIR,
                reps=self.reps,
                size=self.size,
                workers=self.workers,
                basic=basic,
            )

            _pipeline_main(args)

            calcp_files = [
                f for f in os.listdir(workdir) if f.endswith('_GammaP.txt')
            ]
            if not calcp_files:
                raise FileNotFoundError(
                    'BooteJTK/CalcP produced no *_GammaP.txt output in the '
                    f'working directory: {workdir}'
                )
            result_path = os.path.join(workdir, calcp_files[0])
            self.rhythm_results = pd.read_csv(result_path, sep='\t', index_col='ID')
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        return self.rhythm_results

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        pirs_percentile: float = 50,
        tau_threshold: float = 0.5,
        emp_p_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """Assign class labels using pre-computed PIRS and BooteJTK results.

        Call :meth:`run_pirs` and :meth:`run_bootjtk` first, or use
        :meth:`run_all` to do everything in one step.

        Parameters
        ----------
        pirs_percentile : float
            Genes whose PIRS score falls at or below this percentile of all
            scored genes are considered *stable* (default 50).
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
            * ``tau_mean`` – BooteJTK ``TauMean``
            * ``emp_p`` – FDR-corrected p-value (``GammaBH``), if available
            * ``period_mean`` – estimated period, if available
            * ``phase_mean`` – estimated phase, if available
            * ``label`` – one of ``constitutive``, ``rhythmic``,
              ``variable``, ``noisy_rhythmic``
        """
        if self.pirs_scores is None:
            raise RuntimeError("Call run_pirs() before classify().")
        if self.rhythm_results is None:
            raise RuntimeError("Call run_bootjtk() before classify().")

        tau_col = 'TauMean' if 'TauMean' in self.rhythm_results.columns else 'Tau'

        all_ids = self.pirs_scores.index.union(self.rhythm_results.index)
        result = pd.DataFrame(index=all_ids)
        result.index.name = self.pirs_scores.index.name

        result['pirs_score'] = self.pirs_scores['score']
        result['tau_mean'] = self.rhythm_results[tau_col]

        for src_col, dst_col in [
            ('GammaBH', 'emp_p'),
            ('PeriodMean', 'period_mean'),
            ('PhaseMean', 'phase_mean'),
        ]:
            if src_col in self.rhythm_results.columns:
                result[dst_col] = self.rhythm_results[src_col]

        pirs_cutoff = np.percentile(result['pirs_score'].dropna(), pirs_percentile)
        stable = result['pirs_score'] <= pirs_cutoff

        rhythmic = result['tau_mean'] >= tau_threshold
        if 'emp_p' in result.columns:
            rhythmic = rhythmic & (result['emp_p'] <= emp_p_threshold)

        result['label'] = [
            _LABEL_MAP.get((bool(s), bool(r)), 'unclassified')
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
        pirs_alpha: float = 0.5,
        basic: bool = True,
        pirs_percentile: float = 50,
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
        self.run_pirs(alpha=pirs_alpha)
        self.run_bootjtk(basic=basic)
        return self.classify(
            pirs_percentile=pirs_percentile,
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
        means='DEFAULT',
        sds='DEFAULT',
        ns='DEFAULT',
        prefix='classify',
        waveform='cosine',
        period=os.path.join(ref_dir, 'period24.txt'),
        phase=os.path.join(ref_dir, 'phases_00-22_by2.txt'),
        width=os.path.join(ref_dir, 'asymmetries_02-22_by2.txt'),
        output='DEFAULT',
        pickle='DEFAULT',
        id_list='DEFAULT',
        null_list='DEFAULT',
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
        jtk='DEFAULT',
        fit=False,
        null='DEFAULT',
    )
