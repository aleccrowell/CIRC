"""Metric quality tests: each pipeline score should discriminate its target class.

Unlike the structural tests in test_classify.py, these tests run on a clean
high-signal simulation (no batch effects, no missing data) and assert that
continuous scores rank their target class above others.  The intent is to
isolate metric behaviour from pipeline noise so that a failing test here
points directly at the underlying algorithm rather than at imputation/SVA.

Assertions use AUC > 0.5 (ranking) and AP > baseline (precision-recall) so
that they remain valid regardless of classification threshold choices.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, average_precision_score

from circ.simulations import simulate
from circ.expression_classification.classify import Classifier
from circ.pirs.rank import ranker, rsd_ranker


# ---------------------------------------------------------------------------
# Module-scoped fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clean_pipeline(tmp_path_factory):
    """Run Classifier on clean (no batch/missing) simulated data.

    Uses reduced noise (amp_noise=0.4, phase_noise=0.1) and more genes (150)
    than the CI fixture so that AUC estimates are stable and the metrics have
    a fair chance to discriminate their target classes without pipeline noise.
    """
    tmp = tmp_path_factory.mktemp("quality")
    out = str(tmp / "clean.txt")

    sim = simulate(
        tpoints=12, nrows=150, nreps=2, tpoint_space=2,
        pcirc=0.35, plin=0.25,
        phase_noise=0.1, amp_noise=0.4,
        n_batch_effects=0, p_miss=0.0,
        rseed=11,
    )
    sim.write_output(out_name=out)

    true_classes = pd.read_csv(
        out.replace(".txt", "_true_classes.txt"), sep="\t", index_col=0
    )

    clf = Classifier(out, reps=2, size=20)
    result = clf.run_all(slope_pvals=True, n_permutations=99)

    return {"result": result, "true_classes": true_classes}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auc(result, true_classes, score_col, truth_col, invert=False):
    """ROC AUC for score_col vs binary truth_col; None if data is insufficient."""
    shared = result.index.intersection(true_classes.index)
    df = (
        result.loc[shared, [score_col]]
        .join(true_classes.loc[shared, [truth_col]])
        .dropna()
    )
    if len(df) < 10 or df[truth_col].nunique() < 2:
        return None
    scores = -df[score_col].values if invert else df[score_col].values
    return roc_auc_score(df[truth_col].values, scores)


def _ap(result, true_classes, score_col, truth_col, invert=False):
    """Average precision (PR AUC) for score_col vs binary truth_col."""
    shared = result.index.intersection(true_classes.index)
    df = (
        result.loc[shared, [score_col]]
        .join(true_classes.loc[shared, [truth_col]])
        .dropna()
    )
    if len(df) < 10 or df[truth_col].nunique() < 2:
        return None
    scores = -df[score_col].values if invert else df[score_col].values
    return average_precision_score(df[truth_col].values, scores)


def _class_means(result, true_classes, score_col, truth_col):
    """Return (mean score for truth==1, mean score for truth==0)."""
    shared = result.index.intersection(true_classes.index)
    tc = true_classes.loc[shared]
    pos_ids = tc[tc[truth_col] == 1].index
    neg_ids = tc[tc[truth_col] == 0].index
    pos_mean = result.loc[pos_ids, score_col].dropna().mean()
    neg_mean = result.loc[neg_ids, score_col].dropna().mean()
    return pos_mean, neg_mean


# ---------------------------------------------------------------------------
# ROC AUC: each metric should rank its target class above others
# ---------------------------------------------------------------------------

class TestMetricAUC:
    """Each score must achieve AUC > 0.5 against its target class on clean data.

    An AUC below 0.5 on clean (no batch effects, no missing data) data
    indicates a fundamental failure of the underlying metric, not pipeline
    noise.

    PIRS is NOT expected to separate constitutive from circadian — both can be
    PIRS-stable.  PIRS is designed to separate flat (constitutive) from
    variable (linear / noisy) expression.  BooteJTK handles circadian vs
    non-rhythmic discrimination.
    """

    def test_pirs_score_discriminates_linear_from_constitutive(self, clean_pipeline):
        """High PIRS (temporal deviation) should rank linear genes above constitutive.

        This is PIRS's core purpose: a linear ramp creates large prediction-interval
        deviations from mean expression at the time extremes, while flat constitutive
        noise does not.
        """
        result = clean_pipeline["result"]
        tc = clean_pipeline["true_classes"]
        shared = result.index.intersection(tc.index)
        # Restrict to the two classes PIRS is designed to separate
        lin_const_mask = (tc.loc[shared, "Linear"] + tc.loc[shared, "Const"]) == 1
        subset = tc.loc[shared][lin_const_mask]
        merged = result.loc[subset.index, ["pirs_score"]].join(subset[["Linear"]]).dropna()
        if len(merged) < 10 or merged["Linear"].nunique() < 2:
            pytest.skip("insufficient data for AUC test")
        auc = roc_auc_score(merged["Linear"].values, merged["pirs_score"].values)
        assert auc > 0.5, f"PIRS score → Linear (vs Const) AUC={auc:.3f}; expected > 0.5"

    def test_tau_mean_discriminates_circadian(self, clean_pipeline):
        """High TauMean (strong 24h autocorrelation) should rank circadian genes first."""
        auc = _auc(clean_pipeline["result"], clean_pipeline["true_classes"],
                   "tau_mean", "Circadian", invert=False)
        if auc is None:
            pytest.skip("insufficient data for AUC test")
        assert auc > 0.5, f"tau_mean → Circadian AUC={auc:.3f}; expected > 0.5"

    def test_emp_p_discriminates_circadian(self, clean_pipeline):
        """Low GammaBH p-value should rank circadian genes first."""
        auc = _auc(clean_pipeline["result"], clean_pipeline["true_classes"],
                   "emp_p", "Circadian", invert=True)
        if auc is None:
            pytest.skip("insufficient data for AUC test")
        assert auc > 0.5, f"emp_p → Circadian AUC={auc:.3f}; expected > 0.5"

    def test_slope_pval_discriminates_linear(self, clean_pipeline):
        """Low slope p-value should rank linear (ramp) genes first."""
        auc = _auc(clean_pipeline["result"], clean_pipeline["true_classes"],
                   "slope_pval", "Linear", invert=True)
        if auc is None:
            pytest.skip("insufficient data for AUC test")
        assert auc > 0.5, f"slope_pval → Linear AUC={auc:.3f}; expected > 0.5"


# ---------------------------------------------------------------------------
# Average Precision: PR AUC should exceed random baseline
# ---------------------------------------------------------------------------

class TestMetricAP:
    """Average precision must exceed the positive-class prevalence baseline.

    AP == baseline means the score adds no information over random guessing.
    """

    def test_pirs_ap_for_linear_above_baseline(self, clean_pipeline):
        """AP for PIRS discriminating linear from constitutive should exceed prevalence."""
        result = clean_pipeline["result"]
        tc = clean_pipeline["true_classes"]
        shared = result.index.intersection(tc.index)
        lin_const_mask = (tc.loc[shared, "Linear"] + tc.loc[shared, "Const"]) == 1
        subset = tc.loc[shared][lin_const_mask]
        merged = result.loc[subset.index, ["pirs_score"]].join(subset[["Linear"]]).dropna()
        if len(merged) < 10 or merged["Linear"].nunique() < 2:
            pytest.skip("insufficient data")
        ap = average_precision_score(merged["Linear"].values, merged["pirs_score"].values)
        baseline = subset["Linear"].mean()
        assert ap > baseline, f"PIRS AP={ap:.3f} should exceed prevalence baseline ({baseline:.3f})"

    def test_slope_pval_ap_above_baseline(self, clean_pipeline):
        ap = _ap(clean_pipeline["result"], clean_pipeline["true_classes"],
                 "slope_pval", "Linear", invert=True)
        if ap is None:
            pytest.skip("insufficient data")
        baseline = clean_pipeline["true_classes"]["Linear"].mean()
        assert ap > baseline, f"slope_pval AP={ap:.3f} should exceed prevalence baseline ({baseline:.3f})"

    def test_tau_mean_ap_above_baseline(self, clean_pipeline):
        ap = _ap(clean_pipeline["result"], clean_pipeline["true_classes"],
                 "tau_mean", "Circadian", invert=False)
        if ap is None:
            pytest.skip("insufficient data")
        baseline = clean_pipeline["true_classes"]["Circadian"].mean()
        assert ap > baseline, f"tau_mean AP={ap:.3f} should exceed prevalence baseline ({baseline:.3f})"


# ---------------------------------------------------------------------------
# Directional ranking: class means must be in the correct direction
# ---------------------------------------------------------------------------

class TestDirectionalRanking:
    """Mean score values must be directionally consistent with true class membership.

    These are weaker than AUC tests but more interpretable: they tell you
    *which* class is mis-ranked rather than just reporting a combined AUC.
    """

    def test_circadian_has_higher_tau_mean_than_constitutive(self, clean_pipeline):
        pos_mean, neg_mean = _class_means(
            clean_pipeline["result"], clean_pipeline["true_classes"],
            "tau_mean", "Circadian",
        )
        assert pos_mean > neg_mean, (
            f"Circadian tau_mean ({pos_mean:.3f}) should exceed "
            f"non-circadian ({neg_mean:.3f})"
        )

    def test_linear_has_higher_pirs_than_constitutive(self, clean_pipeline):
        """Linear (ramp) genes should have higher PIRS than constitutive (flat noise).

        PIRS measures how far prediction-interval bounds deviate from mean
        expression.  A linear trend pushes the prediction far from the mean at
        time extremes; flat noise does not.
        """
        lin_pirs, _ = _class_means(
            clean_pipeline["result"], clean_pipeline["true_classes"],
            "pirs_score", "Linear",
        )
        const_pirs, _ = _class_means(
            clean_pipeline["result"], clean_pipeline["true_classes"],
            "pirs_score", "Const",
        )
        assert lin_pirs > const_pirs, (
            f"Linear pirs_score ({lin_pirs:.3f}) should exceed "
            f"constitutive ({const_pirs:.3f})"
        )

    def test_linear_has_lower_slope_pval_than_constitutive(self, clean_pipeline):
        lin_pval, const_pval = _class_means(
            clean_pipeline["result"], clean_pipeline["true_classes"],
            "slope_pval", "Linear",
        )
        assert lin_pval < const_pval, (
            f"Linear slope_pval ({lin_pval:.3f}) should be "
            f"lower than constitutive ({const_pval:.3f})"
        )

    def test_circadian_has_lower_emp_p_than_constitutive(self, clean_pipeline):
        circ_empp, const_empp = _class_means(
            clean_pipeline["result"], clean_pipeline["true_classes"],
            "emp_p", "Circadian",
        )
        assert circ_empp < const_empp, (
            f"Circadian emp_p ({circ_empp:.3f}) should be "
            f"lower than constitutive ({const_empp:.3f})"
        )


# ---------------------------------------------------------------------------
# Label accuracy: precision of assigned labels vs ground truth
# ---------------------------------------------------------------------------

class TestLabelAccuracy:
    """Predicted labels should agree with ground truth better than random chance.

    Precision >= class prevalence means the label is informative.  Tests skip
    when fewer than three genes receive a label (e.g. strict emp_p threshold
    with small bootstrap size prevents any rhythmic assignments).
    """

    def _precision_recall(self, result, true_classes, label, truth_col):
        shared = result.index.intersection(true_classes.index)
        pred_pos = result.loc[shared, "label"] == label
        act_pos = true_classes.loc[shared, truth_col] == 1
        tp = (pred_pos & act_pos).sum()
        fp = (pred_pos & ~act_pos).sum()
        fn = (~pred_pos & act_pos).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return precision, recall

    def test_compound_rhythmic_precision_above_chance(self, clean_pipeline):
        """Genes labeled rhythmic OR noisy_rhythmic should be enriched for true circadian genes.

        PIRS determines whether a gene is stable or variable; BooteJTK determines
        whether it is rhythmic.  Truly circadian genes can be either stable
        (→ 'rhythmic') or variable (→ 'noisy_rhythmic') depending on their noise
        level, so the compound label set is the correct thing to check.
        """
        result = clean_pipeline["result"]
        tc = clean_pipeline["true_classes"]
        shared = result.index.intersection(tc.index)
        rhythmic_labels = {"rhythmic", "noisy_rhythmic"}
        n_detected = result.loc[shared, "label"].isin(rhythmic_labels).sum()
        if n_detected < 3:
            pytest.skip("too few rhythmic/noisy_rhythmic labels for meaningful precision test")
        pred_pos = result.loc[shared, "label"].isin(rhythmic_labels)
        act_pos = tc.loc[shared, "Circadian"] == 1
        tp = (pred_pos & act_pos).sum()
        fp = (pred_pos & ~act_pos).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        baseline = tc["Circadian"].mean()
        assert precision >= baseline, (
            f"rhythmic+noisy_rhythmic precision ({precision:.3f}) should exceed "
            f"prevalence ({baseline:.3f})"
        )

    def test_constitutive_label_precision_above_chance(self, clean_pipeline):
        result = clean_pipeline["result"]
        tc = clean_pipeline["true_classes"]
        if (result["label"] == "constitutive").sum() < 3:
            pytest.skip("too few 'constitutive' labels")
        precision, _ = self._precision_recall(result, tc, "constitutive", "Const")
        baseline = tc["Const"].mean()
        assert precision >= baseline, (
            f"'constitutive' precision ({precision:.3f}) should exceed "
            f"prevalence ({baseline:.3f})"
        )

    def test_linear_label_precision_above_chance(self, clean_pipeline):
        result = clean_pipeline["result"]
        tc = clean_pipeline["true_classes"]
        if (result["label"] == "linear").sum() < 3:
            pytest.skip("too few 'linear' labels")
        precision, _ = self._precision_recall(result, tc, "linear", "Linear")
        baseline = tc["Linear"].mean()
        assert precision >= baseline, (
            f"'linear' precision ({precision:.3f}) should exceed "
            f"prevalence ({baseline:.3f})"
        )

    def test_at_least_one_circadian_gene_labelled_rhythmic_or_noisy_rhythmic(
        self, clean_pipeline
    ):
        """On clean data, at least one truly circadian gene should be detected as rhythmic."""
        result = clean_pipeline["result"]
        tc = clean_pipeline["true_classes"]
        shared = result.index.intersection(tc.index)
        circ_ids = tc.loc[shared][tc.loc[shared]["Circadian"] == 1].index
        if len(circ_ids) < 5:
            pytest.skip("too few circadian genes")
        rhythmic_labels = {"rhythmic", "noisy_rhythmic"}
        n_detected = result.loc[circ_ids, "label"].isin(rhythmic_labels).sum()
        assert n_detected >= 1, (
            f"Expected at least 1 truly circadian gene labelled rhythmic/noisy_rhythmic "
            f"on clean data; got 0 out of {len(circ_ids)}"
        )


# ---------------------------------------------------------------------------
# PIRS noise discrimination: flat high-noise vs flat low-noise constitutive
# ---------------------------------------------------------------------------

class TestPirsNoiseDiscrimination:
    """PIRS score discriminates flat high-noise from flat low-noise constitutive genes.

    Both gene types are constitutive (no slope, no rhythm), but PIRS must
    assign higher scores to high-noise genes because wider prediction intervals
    deviate further from mean expression.  Neither gene should be flagged as
    non-constitutive or trending by the permutation tests.
    """

    @pytest.fixture(scope="class")
    def flat_noise_ranker(self, tmp_path_factory):
        """One low-noise (σ=0.1) and one high-noise (σ=2.0) flat gene."""
        tmp = tmp_path_factory.mktemp("flat_noise")
        rng = np.random.default_rng(42)
        tpoints = [2, 4, 6, 8, 10, 12]
        n_reps = 3
        cols = [f"CT{t:02d}_{r}" for t in tpoints for r in range(1, n_reps + 1)]
        flat_low  = rng.normal(5.0, 0.1, len(cols))
        flat_high = rng.normal(5.0, 2.0, len(cols))
        df = pd.DataFrame({"flat_low": flat_low, "flat_high": flat_high}, index=cols).T
        df.index.name = "#"
        path = str(tmp / "flat_noise.txt")
        df.to_csv(path, sep="\t")
        r = ranker(path, anova=False)
        r.get_tpoints()
        r.calculate_scores()
        return r

    def test_high_noise_scores_higher(self, flat_noise_ranker):
        """Wider prediction intervals from more noise → larger PIRS score."""
        scores = flat_noise_ranker.errors
        assert scores.loc["flat_high", "score"] > scores.loc["flat_low", "score"]

    def test_low_noise_ranked_ahead_of_high_noise(self, flat_noise_ranker):
        """Low-noise gene sorts before high-noise gene (better constitutive marker)."""
        ranked = list(flat_noise_ranker.errors.index)
        assert ranked.index("flat_low") < ranked.index("flat_high")

    def test_neither_gene_flagged_as_nonconstit(self, flat_noise_ranker):
        """Both flat genes should have high PIRS p-values regardless of noise level."""
        result = flat_noise_ranker.calculate_pvals(n_permutations=199)
        assert result.loc["flat_low",  "pval"] > 0.1
        assert result.loc["flat_high", "pval"] > 0.1

    def test_neither_gene_has_significant_slope(self, flat_noise_ranker):
        """Neither flat gene should be detected as significantly trending."""
        result = flat_noise_ranker.calculate_slope_pvals(n_permutations=199)
        assert result.loc["flat_low",  "slope_pval"] > 0.1
        assert result.loc["flat_high", "slope_pval"] > 0.1

    def test_rsd_also_distinguishes_noise_levels(self, tmp_path_factory):
        """rsd_ranker independently confirms the noise-level ranking."""
        tmp = tmp_path_factory.mktemp("flat_noise_rsd")
        rng = np.random.default_rng(42)
        tpoints = [2, 4, 6, 8, 10, 12]
        n_reps = 3
        cols = [f"CT{t:02d}_{r}" for t in tpoints for r in range(1, n_reps + 1)]
        flat_low  = rng.normal(5.0, 0.1, len(cols))
        flat_high = rng.normal(5.0, 2.0, len(cols))
        df = pd.DataFrame({"flat_low": flat_low, "flat_high": flat_high}, index=cols).T
        df.index.name = "#"
        path = str(tmp / "flat_noise_rsd.txt")
        df.to_csv(path, sep="\t")
        scores = rsd_ranker(path).calculate_scores()
        assert scores.loc["flat_low", "score"] < scores.loc["flat_high", "score"]
