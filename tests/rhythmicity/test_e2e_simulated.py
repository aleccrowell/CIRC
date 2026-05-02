"""End-to-end tests for BooteJTK using synthetically generated time-series.

Tests exercise the full programmatic pipeline:
  - BooteJTK: output file created with correct schema
  - BooteJTK: rhythmic genes get high TauMean, flat genes get lower TauMean
  - BooteJTK: phase and period are recovered accurately
  - CalcP: valid p-values produced; rhythmic genes are significant
"""

import argparse
import os
import numpy as np
import pandas as pd
import pytest

from circ.rhythmicity import BooteJTK, CalcP
from circ.rhythmicity.BooteJTK import _REF_DIR


# ── simulation parameters ─────────────────────────────────────────────────

PERIOD = 24
N_TP = 12  # unique ZT timepoints per cycle (ZT0, ZT2, …, ZT22)
AMPLITUDE = 2.0  # peak-to-mean signal amplitude (SNR ≈ 40 vs noise)
NOISE_SD = 0.05  # per-observation noise for rhythmic genes
FLAT_SD = 0.30  # noise for flat/arrhythmic genes
BOOTSTRAPS = 30  # small enough to be fast, large enough to be stable
N_NULL = 100  # white-noise genes used to build the null distribution

# Phase search grid is 0–22 by 2 h, so use multiples of 2 for exact recovery.
TRUE_PHASES = (0, 6, 12, 18)


# ── data-generation helpers ───────────────────────────────────────────────


def _zt_header():
    """Return (time_array, label_list) covering two full 24-h cycles (24 cols)."""
    times = np.arange(N_TP) * (PERIOD / N_TP)  # 0, 2, 4, …, 22
    full = np.concatenate([times, times + PERIOD])  # 0..22, 24..46
    return full, [f"ZT{int(t)}" for t in full]


def write_simulated_data(path, rng):
    """Write a tab-delimited input file with rhythmic and flat genes.

    Returns a dict mapping gene_id → true_phase_hours (None for flat genes).
    """
    full_times, labels = _zt_header()
    gene_info = {}
    rows = []

    for ph in TRUE_PHASES:
        gid = f"rhythm_{ph:02d}h"
        sig = AMPLITUDE * np.cos(2 * np.pi * (full_times - ph) / PERIOD) + 10.0
        vals = sig + rng.normal(0, NOISE_SD, len(full_times))
        rows.append((gid, vals))
        gene_info[gid] = ph

    for i in range(4):
        gid = f"flat_{i}"
        vals = rng.normal(10.0, FLAT_SD, len(full_times))
        rows.append((gid, vals))
        gene_info[gid] = None

    with open(path, "w") as fh:
        fh.write("#\t" + "\t".join(labels) + "\n")
        for gid, vals in rows:
            fh.write(gid + "\t" + "\t".join(f"{v:.6f}" for v in vals) + "\n")
    return gene_info


def write_null_data(path, n_genes=N_NULL):
    """Write white-noise genes to use as the null distribution for CalcP."""
    _, labels = _zt_header()
    rng = np.random.default_rng(999)
    with open(path, "w") as fh:
        fh.write("#\t" + "\t".join(labels) + "\n")
        for i in range(n_genes):
            vals = rng.normal(0, 1, len(labels))
            fh.write(f"null_{i}\t" + "\t".join(f"{v:.6f}" for v in vals) + "\n")


# ── argparse helpers ──────────────────────────────────────────────────────


def _bootejtk_args(data_file, size=BOOTSTRAPS):
    """Minimal Namespace accepted by BooteJTK.main()."""
    return argparse.Namespace(
        filename=data_file,
        means="DEFAULT",
        sds="DEFAULT",
        ns="DEFAULT",
        prefix="e2e",
        waveform="cosine",
        period=os.path.join(_REF_DIR, "period24.txt"),
        phase=os.path.join(_REF_DIR, "phases_00-22_by2.txt"),
        width=os.path.join(_REF_DIR, "asymmetries_02-22_by2.txt"),
        output="DEFAULT",
        pickle="DEFAULT",
        id_list="DEFAULT",
        null_list="DEFAULT",
        size=size,
        reps=2,
        write=False,
        workers=1,
        harding=False,
        normal=False,
    )


def _calcp_args(signal_out, null_out):
    """Minimal Namespace accepted by CalcP.main()."""
    return argparse.Namespace(filename=signal_out, null=null_out, fit=False)


# ── module-scoped fixtures ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bootejtk_results(tmp_path_factory):
    """Run BooteJTK once on simulated data; return (DataFrame, gene_info)."""
    d = tmp_path_factory.mktemp("e2e_bootjtk")
    rng = np.random.default_rng(0)
    data_path = str(d / "sim_data.txt")
    gene_info = write_simulated_data(data_path, rng)
    fn_out, _, _ = BooteJTK.main(_bootejtk_args(data_path))
    df = pd.read_table(fn_out)
    return df, gene_info


@pytest.fixture(scope="module")
def calcp_results(tmp_path_factory):
    """Run BooteJTK + CalcP on signal and null; return (GammaP DataFrame, gene_info)."""
    d = tmp_path_factory.mktemp("e2e_calcp")
    rng = np.random.default_rng(1)

    # signal run
    data_path = str(d / "sig_data.txt")
    gene_info = write_simulated_data(data_path, rng)
    sig_out, _, _ = BooteJTK.main(_bootejtk_args(data_path))

    # null run
    null_path = str(d / "null_data.txt")
    write_null_data(null_path)
    null_out, _, _ = BooteJTK.main(_bootejtk_args(null_path))

    # p-value calculation
    CalcP.main(_calcp_args(sig_out, null_out))
    gammap_path = sig_out.replace(".txt", "_GammaP.txt")
    df = pd.read_table(gammap_path, index_col="ID")
    return df, gene_info


# ── BooteJTK output structure ─────────────────────────────────────────────


class TestBooteJTKOutputStructure:
    def test_correct_row_count(self, bootejtk_results):
        df, gene_info = bootejtk_results
        assert len(df) == len(gene_info)

    def test_required_columns_present(self, bootejtk_results):
        df, _ = bootejtk_results
        required = {
            "ID",
            "Waveform",
            "TauMean",
            "TauStdDev",
            "PhaseMean",
            "PhaseStdDev",
            "PeriodMean",
        }
        assert required.issubset(df.columns)

    def test_all_gene_ids_in_output(self, bootejtk_results):
        df, gene_info = bootejtk_results
        assert set(df["ID"]) == set(gene_info.keys())

    def test_tau_values_in_valid_range(self, bootejtk_results):
        # TauMean is arctanh-transformed (Fisher Z), so range is ~(-2.65, 2.65)
        # corresponding to clipped raw tau in [-0.99, 0.99]
        max_z = float(np.arctanh(0.99))
        df, _ = bootejtk_results
        assert df["TauMean"].between(-max_z, max_z).all()

    def test_phase_values_within_period(self, bootejtk_results):
        df, _ = bootejtk_results
        assert df["PhaseMean"].between(0, PERIOD, inclusive="left").all()

    def test_period_values_are_positive(self, bootejtk_results):
        df, _ = bootejtk_results
        assert (df["PeriodMean"] > 0).all()

    def test_numeric_columns_finite(self, bootejtk_results):
        df, _ = bootejtk_results
        for col in ("TauMean", "TauStdDev", "PhaseMean", "PeriodMean"):
            assert df[col].notna().all(), f"NaN found in column '{col}'"


# ── rhythm detection accuracy ─────────────────────────────────────────────


class TestRhythmDetection:
    def test_rhythmic_genes_have_high_tau(self, bootejtk_results):
        """Strong cosine signal (SNR ≈ 40) should produce TauMean > 0.5.

        TauMean is arctanh-transformed, so 0.5 on this scale corresponds to
        a raw Kendall tau of ~0.46 — a conservative lower bound.
        """
        df, gene_info = bootejtk_results
        df = df.set_index("ID")
        for gid, ph in gene_info.items():
            if ph is not None:
                tau = df.loc[gid, "TauMean"]
                assert tau > 0.5, f"{gid}: TauMean={tau:.3f}, expected > 0.5"

    def test_rhythmic_beats_flat_on_average(self, bootejtk_results):
        df, gene_info = bootejtk_results
        df = df.set_index("ID")
        rhythmic_tau = df.loc[
            [g for g, p in gene_info.items() if p is not None], "TauMean"
        ]
        flat_tau = df.loc[[g for g, p in gene_info.items() if p is None], "TauMean"]
        assert rhythmic_tau.mean() > flat_tau.mean()

    def test_period_recovered_as_24h(self, bootejtk_results):
        df, gene_info = bootejtk_results
        df = df.set_index("ID")
        for gid, ph in gene_info.items():
            if ph is not None:
                period = df.loc[gid, "PeriodMean"]
                assert abs(period - PERIOD) <= 2, (
                    f"{gid}: PeriodMean={period}, expected ~{PERIOD}"
                )

    def test_phase_recovered_within_4h(self, bootejtk_results):
        """Detected phase should be within 4 hours of truth (2 grid steps)."""
        df, gene_info = bootejtk_results
        df = df.set_index("ID")
        for gid, true_ph in gene_info.items():
            if true_ph is None:
                continue
            det = df.loc[gid, "PhaseMean"]
            circ_diff = abs((det - true_ph + PERIOD / 2) % PERIOD - PERIOD / 2)
            assert circ_diff <= 4, (
                f"{gid}: true_phase={true_ph}h, detected={det:.1f}h "
                f"(circular diff={circ_diff:.1f}h)"
            )

    def test_different_phases_produce_different_estimates(self, bootejtk_results):
        """Genes with phases 6 h apart should be distinguished."""
        df, gene_info = bootejtk_results
        df = df.set_index("ID")
        phase_0 = df.loc["rhythm_00h", "PhaseMean"]
        phase_12 = df.loc["rhythm_12h", "PhaseMean"]
        circ_diff = abs((phase_0 - phase_12 + PERIOD / 2) % PERIOD - PERIOD / 2)
        assert circ_diff >= 6, (
            f"rhythm_00h phase={phase_0:.1f}h and rhythm_12h phase={phase_12:.1f}h "
            f"should be ≥6h apart, got {circ_diff:.1f}h"
        )


# ── CalcP / full pipeline tests ────────────────────────────────────────────


class TestCalcPOutput:
    def test_correct_row_count(self, calcp_results):
        df, gene_info = calcp_results
        assert len(df) == len(gene_info)

    def test_pvalue_columns_present(self, calcp_results):
        df, _ = calcp_results
        for col in ("empP", "GammaP", "GammaBH"):
            assert col in df.columns

    def test_pvalues_in_unit_interval(self, calcp_results):
        df, _ = calcp_results
        for col in ("empP", "GammaP", "GammaBH"):
            out_of_range = df[col][~df[col].between(0, 1)]
            assert out_of_range.empty, (
                f"{col} has out-of-range values: {out_of_range.tolist()}"
            )

    def test_gammabh_geq_gammap(self, calcp_results):
        """FDR-corrected values should be ≥ the uncorrected p-values."""
        df, _ = calcp_results
        assert (df["GammaBH"] >= df["GammaP"] - 1e-9).all()

    def test_rhythmic_genes_lower_pvalue_than_flat(self, calcp_results):
        df, gene_info = calcp_results
        r_ids = [g for g, p in gene_info.items() if p is not None]
        f_ids = [g for g, p in gene_info.items() if p is None]
        assert df.loc[r_ids, "GammaP"].mean() < df.loc[f_ids, "GammaP"].mean()

    def test_rhythmic_genes_significant_at_alpha_05(self, calcp_results):
        """With amplitude=2 and 100 null genes, rhythmic genes must pass α=0.05."""
        df, gene_info = calcp_results
        for gid, ph in gene_info.items():
            if ph is not None:
                p = df.loc[gid, "GammaP"]
                assert p < 0.05, f"{gid}: GammaP={p:.4f}, expected < 0.05"

    def test_flat_genes_have_higher_pvalues_than_rhythmic(self, calcp_results):
        df, gene_info = calcp_results
        r_max_p = max(
            df.loc[g, "GammaP"] for g, p in gene_info.items() if p is not None
        )
        f_min_p = min(df.loc[g, "GammaP"] for g, p in gene_info.items() if p is None)
        assert r_max_p < f_min_p, (
            f"All rhythmic GammaP values ({r_max_p:.4f}) should be below "
            f"all flat GammaP values ({f_min_p:.4f})"
        )
