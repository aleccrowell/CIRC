import os
import random
import string

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale


class simulate:
    """
    Generates a simulated time-series dataset with three expression classes.

    Rows are drawn independently from circadian (sinusoidal), linear (ramp),
    and constitutive (flat) distributions.  Optional batch effects and missing
    data can be layered on top to support proteomics-style benchmarking.

    Parameters
    ----------
    tpoints : int
        Number of time points.
    nrows : int
        Number of expression profiles (genes / peptides).
    nreps : int
        Biological replicates per time point.
    tpoint_space : int
        Hours between time points (ZT step size).
    pcirc : float
        Fraction of profiles that are circadian.
    plin : float
        Fraction of profiles that follow a linear trend.
        The remaining fraction (1 - pcirc - plin) are constitutive.
    phase_prop : float
        Fraction of circadian profiles in the secondary phase (phase + π).
    phase_noise : float
        Standard deviation of per-profile phase jitter.
    amp_noise : float
        Standard deviation of amplitude noise added to every profile.
    n_batch_effects : int
        Number of distinct batch effects to simulate (0 = none).
    pbatch : float
        Per-batch probability that a given profile is affected.
    effect_size : float
        SD of the normal distribution used to draw each batch effect vector.
    p_miss : float
        Probability that a row contains any missing values (0 = none).
    lam_miss : float
        Poisson λ for the number of missing columns per affected row.
    rseed : int
        Random seed for reproducibility.

    Attributes
    ----------
    cols : list of str
        Sample column names in ZT{HH}_{rep} format.
    classes : numpy.ndarray of str
        Per-row class label: ``"circadian"``, ``"linear"``, or
        ``"constitutive"``.
    sim : numpy.ndarray, shape (nrows, tpoints * nreps)
        Clean simulated data, row-scaled to unit standard deviation.
    sim_miss : numpy.ndarray, shape (nrows, tpoints * nreps)
        Simulated data with batch effects and missing values applied.
        Missing entries are represented as ``NaN``.
    """

    def __init__(
        self,
        tpoints=24,
        nrows=1000,
        nreps=3,
        tpoint_space=2,
        pcirc=0.3,
        plin=0.2,
        phase_prop=0.5,
        phase_noise=0.25,
        amp_noise=0.75,
        n_batch_effects=0,
        pbatch=0.5,
        effect_size=2.0,
        p_miss=0.0,
        lam_miss=5,
        rseed=0,
    ):
        pconst = 1.0 - pcirc - plin
        if pconst < 0:
            raise ValueError("pcirc + plin must be <= 1.0")

        np.random.seed(rseed)
        random.seed(rseed)

        self.tpoints = int(tpoints)
        self.nreps = int(nreps)
        self.nrows = int(nrows)
        self.tpoint_space = int(tpoint_space)

        # Column names: ZT{HH}_{rep}
        self.cols = [
            "ZT{:02d}_{}".format(tpoint_space * i + tpoint_space, j + 1)
            for i in range(self.tpoints)
            for j in range(self.nreps)
        ]
        n_cols = self.tpoints * self.nreps

        # Draw class labels
        self.classes = np.random.choice(
            ["circadian", "linear", "constitutive"],
            size=self.nrows,
            p=[pcirc, plin, pconst],
        )

        base = 2 * np.pi * np.arange(1, self.tpoints + 1) * self.tpoint_space / 24
        ramp = np.linspace(0, 2, self.tpoints)

        # Build clean simulation matrix
        raw = np.empty((self.nrows, n_cols), dtype=float)
        for idx, cls in enumerate(self.classes):
            if cls == "circadian":
                p = np.random.binomial(1, phase_prop)
                reps = [
                    np.sin(base + np.random.normal(0, phase_noise) + np.pi * p)
                    + np.random.normal(0, amp_noise, self.tpoints)
                    for _ in range(self.nreps)
                ]
                raw[idx] = [v for t in zip(*reps) for v in t]
            elif cls == "linear":
                reps = [
                    ramp + np.random.normal(0, amp_noise, self.tpoints)
                    for _ in range(self.nreps)
                ]
                raw[idx] = [v for t in zip(*reps) for v in t]
            else:  # constitutive
                raw[idx] = np.random.normal(1, amp_noise, n_cols)

        self._raw = raw
        # Row-scale to unit standard deviation for expression analysis
        self.sim = scale(raw, axis=1, with_mean=False)

        # Batch effects applied to unscaled data
        noisy = raw.copy()
        if n_batch_effects > 0:
            batch_vectors = [
                np.random.normal(0.0, effect_size, n_cols)
                for _ in range(n_batch_effects)
            ]
            hits = np.random.binomial(1, pbatch, (self.nrows, n_batch_effects))
            for k, bv in enumerate(batch_vectors):
                noisy += hits[:, k : k + 1] * bv

        # Missing data mask
        self.sim_miss = noisy.copy()
        miss_rows = np.where(np.random.binomial(1, p_miss, self.nrows))[0]
        for i in miss_rows:
            num_miss = min(np.random.poisson(lam_miss) + 1, n_cols)
            cols_missing = np.random.choice(n_cols, size=num_miss, replace=False)
            self.sim_miss[i, cols_missing] = np.nan

    # ------------------------------------------------------------------
    # Backward-compatible class-label properties
    # ------------------------------------------------------------------

    @property
    def circ(self):
        """int array, 1 where class == 'circadian'."""
        return (self.classes == "circadian").astype(int)

    @property
    def const(self):
        """int array, 1 where class == 'constitutive'."""
        return (self.classes == "constitutive").astype(int)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _true_classes_df(self, index=None):
        df = pd.DataFrame(
            {
                "Circadian": (self.classes == "circadian").astype(int),
                "Linear": (self.classes == "linear").astype(int),
                "Const": (self.classes == "constitutive").astype(int),
            }
        )
        if index is not None:
            df.index = index
        return df

    # ------------------------------------------------------------------
    # Output methods
    # ------------------------------------------------------------------

    def write_output(self, out_name="simulated_data.txt"):
        """Write expression-format output (gene / # index).

        Writes the scaled clean data to *out_name* and the per-row class
        labels to ``<stem>_true_classes.txt`` (where stem is *out_name*
        with its final four characters removed).
        """
        df = pd.DataFrame(self.sim, columns=self.cols)
        df.index.name = "#"
        df.to_csv(out_name, sep="\t")
        stem = out_name[:-4]
        self._true_classes_df(index=df.index).to_csv(
            stem + "_true_classes.txt", sep="\t"
        )

    def write_proteomics(self, out_name="simulated_data"):
        """Write proteomics-format output (Peptide / Protein index).

        Writes three files:

        - ``<out_name>_with_noise.txt``: batch-effect and missing-data
          version with Peptide/Protein columns; missing entries written as
          ``NULL``.
        - ``<out_name>_baseline.txt``: clean unscaled data with Protein
          index.
        - ``<out_name>_true_classes.txt``: class labels indexed by Protein.
        """
        peps = [
            "".join(random.choices(string.ascii_uppercase, k=12))
            for _ in range(self.nrows)
        ]
        prots = [
            "".join(random.choices(string.ascii_uppercase, k=12))
            for _ in range(self.nrows)
        ]

        # With-noise file (batch effects + missing → NULL)
        noisy_df = pd.DataFrame(self.sim_miss, columns=self.cols).fillna("NULL")
        noisy_df.insert(0, "Protein", prots)
        noisy_df.insert(0, "Peptide", peps)
        noisy_df.set_index("Peptide", inplace=True)
        noisy_df.insert(len(self.cols) + 1, "pool_01", ["1"] * self.nrows)
        noisy_df.to_csv(out_name + "_with_noise.txt", sep="\t")

        # Baseline file (clean, unscaled)
        base_df = pd.DataFrame(self._raw, columns=self.cols)
        base_df.insert(0, "Protein", prots)
        base_df.set_index("Protein", inplace=True)
        base_df.index.name = "#"
        base_df.to_csv(out_name + "_baseline.txt", sep="\t")

        # True classes indexed by Protein
        self._true_classes_df(
            index=pd.Index(prots, name="Protein")
        ).to_csv(out_name + "_true_classes.txt", sep="\t")

    def generate_pool_map(self, out_name="pool_map"):
        """Write a pool map parquet file mapping every sample column to pool 1."""
        pd.DataFrame({"pool_number": {col: 1 for col in self.cols}}).to_parquet(
            out_name + ".parquet"
        )

    def write_genorm(self, out_name="simulated_data_genorm.txt"):
        """Write geNorm-format output (Sample, Detector, Cq columns)."""
        df = pd.DataFrame(self.sim, columns=self.cols)
        df.index.name = "#"
        long = df.reset_index().melt(id_vars=["#"])
        long.columns = ["Detector", "Sample", "Cq"]
        long = long[["Sample", "Detector", "Cq"]]
        long["Sample"] = long["Sample"].apply(
            lambda x: "timepoint_" + str(x).split("_")[0]
        )
        long["Detector"] = long["Detector"].astype(str).apply(
            lambda x: "gene_" + x
        )
        long["Cq"] = (long["Cq"] - long["Cq"].min()) / (
            long["Cq"].max() - long["Cq"].min()
        )
        long.to_csv(out_name, sep=" ", index=False)

    def write_normfinder(self, out_name="simulated_data_normfinder.txt"):
        """Write NormFinder-format output (Sample, Detector, Cq columns)."""
        df = pd.DataFrame(self.sim, columns=self.cols)
        df.index.name = "#"
        long = df.reset_index().melt(id_vars=["#"])
        long.columns = ["Detector", "Sample", "Cq"]
        long = long[["Sample", "Detector", "Cq"]]
        long["Sample"] = long["Sample"].astype(str).apply(
            lambda x: "timepoint_" + x
        )
        long["Detector"] = long["Detector"].astype(str).apply(
            lambda x: "gene_" + x
        )
        long["Cq"] = (long["Cq"] - long["Cq"].min()) / (
            long["Cq"].max() - long["Cq"].min()
        )
        long.to_csv(out_name, sep=" ", index=False)
