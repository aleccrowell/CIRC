"""Tests for the unified circ.simulations.simulate class."""

import os
import numpy as np
import pandas as pd
import pytest

from circ.simulations import simulate


# ---------------------------------------------------------------------------
# Construction and core attributes
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_instantiation(self):
        sim = simulate()
        assert sim is not None

    def test_sim_shape(self):
        sim = simulate(tpoints=4, nrows=20, nreps=2, tpoint_space=2)
        assert sim.sim.shape == (20, 4 * 2)

    def test_sim_miss_shape(self):
        sim = simulate(tpoints=4, nrows=20, nreps=2, tpoint_space=2)
        assert sim.sim_miss.shape == (20, 4 * 2)

    def test_column_count(self):
        sim = simulate(tpoints=6, nrows=10, nreps=3, tpoint_space=2)
        assert len(sim.cols) == 6 * 3

    def test_column_name_format(self):
        sim = simulate(tpoints=4, nrows=10, nreps=2, tpoint_space=2)
        assert sim.cols[0] == "ZT02_1"
        assert sim.cols[1] == "ZT02_2"
        assert sim.cols[2] == "ZT04_1"

    def test_column_name_custom_spacing(self):
        sim = simulate(tpoints=4, nrows=10, nreps=2, tpoint_space=4)
        assert sim.cols[0] == "ZT04_1"
        assert sim.cols[2] == "ZT08_1"

    def test_three_digit_column_names(self):
        # tpoint_space=4, 25 tpoints → last time = 4*25 = 100 → ZT100
        sim = simulate(tpoints=25, nrows=10, nreps=2, tpoint_space=4)
        assert "ZT100_1" in sim.cols
        assert "ZT100_2" in sim.cols

    def test_sim_row_scaled(self):
        sim = simulate(nrows=50, rseed=7)
        stds = np.std(sim.sim, axis=1)
        np.testing.assert_allclose(stds, np.ones(50), atol=1e-10)

    def test_reproducibility(self):
        sim1 = simulate(nrows=50, rseed=42)
        sim2 = simulate(nrows=50, rseed=42)
        np.testing.assert_array_equal(sim1.sim, sim2.sim)
        np.testing.assert_array_equal(sim1.classes, sim2.classes)

    def test_different_seeds_differ(self):
        sim1 = simulate(nrows=50, rseed=0)
        sim2 = simulate(nrows=50, rseed=99)
        assert not np.array_equal(sim1.sim, sim2.sim)

    def test_invalid_proportions_raises(self):
        with pytest.raises(ValueError):
            simulate(pcirc=0.7, plin=0.7)


# ---------------------------------------------------------------------------
# Three-class labels
# ---------------------------------------------------------------------------


class TestClassLabels:
    def test_classes_array_length(self):
        sim = simulate(nrows=100)
        assert len(sim.classes) == 100

    def test_classes_only_valid_values(self):
        sim = simulate(nrows=200, rseed=0)
        assert set(sim.classes).issubset({"circadian", "linear", "constitutive"})

    def test_circ_property_binary(self):
        sim = simulate(nrows=100)
        assert set(np.unique(sim.circ)).issubset({0, 1})

    def test_const_property_binary(self):
        sim = simulate(nrows=100)
        assert set(np.unique(sim.const)).issubset({0, 1})

    def test_circ_matches_classes(self):
        sim = simulate(nrows=200, rseed=1)
        np.testing.assert_array_equal(
            sim.circ, (sim.classes == "circadian").astype(int)
        )

    def test_const_matches_classes(self):
        sim = simulate(nrows=200, rseed=1)
        np.testing.assert_array_equal(
            sim.const, (sim.classes == "constitutive").astype(int)
        )

    def test_pcirc_controls_circadian_fraction(self):
        sim_high = simulate(nrows=5000, pcirc=0.8, plin=0.1, rseed=1)
        sim_low = simulate(nrows=5000, pcirc=0.1, plin=0.1, rseed=1)
        assert sim_high.circ.mean() > sim_low.circ.mean()

    def test_plin_controls_linear_fraction(self):
        sim_high = simulate(nrows=5000, pcirc=0.1, plin=0.8, rseed=2)
        sim_low = simulate(nrows=5000, pcirc=0.1, plin=0.1, rseed=2)
        linear_high = np.mean(sim_high.classes == "linear")
        linear_low = np.mean(sim_low.classes == "linear")
        assert linear_high > linear_low

    def test_proportions_approximately_correct(self):
        sim = simulate(nrows=10000, pcirc=0.3, plin=0.2, rseed=0)
        assert 0.20 < sim.circ.mean() < 0.40
        assert 0.10 < np.mean(sim.classes == "linear") < 0.30
        assert 0.40 < sim.const.mean() < 0.60

    def test_all_constitutive_when_pcirc_and_plin_zero(self):
        sim = simulate(nrows=100, pcirc=0.0, plin=0.0, rseed=0)
        assert (sim.classes == "constitutive").all()

    def test_three_classes_partition_rows(self):
        sim = simulate(nrows=200, rseed=5)
        circ = sim.circ
        const = sim.const
        linear = (sim.classes == "linear").astype(int)
        np.testing.assert_array_equal(circ + const + linear, np.ones(200, dtype=int))


# ---------------------------------------------------------------------------
# Batch effects
# ---------------------------------------------------------------------------


class TestBatchEffects:
    def test_no_batch_effects_by_default(self):
        sim = simulate(nrows=50, rseed=0)
        # With no batch effects, sim_miss should equal _raw
        np.testing.assert_array_equal(sim.sim_miss, sim._raw)

    def test_batch_effects_change_data(self):
        sim_clean = simulate(nrows=50, n_batch_effects=0, rseed=0)
        sim_noisy = simulate(nrows=50, n_batch_effects=3, pbatch=1.0, rseed=0)
        assert not np.allclose(sim_clean.sim_miss, sim_noisy.sim_miss)

    def test_batch_effect_reproducible(self):
        sim1 = simulate(nrows=30, n_batch_effects=2, rseed=7)
        sim2 = simulate(nrows=30, n_batch_effects=2, rseed=7)
        np.testing.assert_array_equal(sim1.sim_miss, sim2.sim_miss)

    def test_zero_pbatch_means_no_effect(self):
        sim_clean = simulate(nrows=50, n_batch_effects=0, rseed=0)
        sim_zero = simulate(nrows=50, n_batch_effects=5, pbatch=0.0, rseed=0)
        np.testing.assert_array_equal(sim_clean.sim_miss, sim_zero.sim_miss)


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------


class TestMissingData:
    def test_no_missing_by_default(self):
        sim = simulate(nrows=50, rseed=0)
        assert not np.isnan(sim.sim_miss).any()

    def test_missing_data_produces_nans(self):
        sim = simulate(nrows=100, p_miss=0.5, rseed=3)
        assert np.isnan(sim.sim_miss).any()

    def test_missing_fraction_controlled_by_p_miss(self):
        sim_low = simulate(nrows=500, p_miss=0.1, rseed=4)
        sim_high = simulate(nrows=500, p_miss=0.9, rseed=4)
        rows_with_nan_low = np.isnan(sim_low.sim_miss).any(axis=1).mean()
        rows_with_nan_high = np.isnan(sim_high.sim_miss).any(axis=1).mean()
        assert rows_with_nan_low < rows_with_nan_high

    def test_missing_reproducible(self):
        sim1 = simulate(nrows=30, p_miss=0.3, rseed=9)
        sim2 = simulate(nrows=30, p_miss=0.3, rseed=9)
        np.testing.assert_array_equal(np.isnan(sim1.sim_miss), np.isnan(sim2.sim_miss))


# ---------------------------------------------------------------------------
# write_output (expression format)
# ---------------------------------------------------------------------------


class TestWriteOutput:
    def test_creates_data_file(self, tmp_path):
        sim = simulate(nrows=20, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "sim_out.txt")
        sim.write_output(out_name=out)
        assert os.path.exists(out)

    def test_creates_true_classes_file(self, tmp_path):
        sim = simulate(nrows=20, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "sim_out.txt")
        sim.write_output(out_name=out)
        assert os.path.exists(str(tmp_path / "sim_out_true_classes.txt"))

    def test_data_shape(self, tmp_path):
        sim = simulate(nrows=20, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "sim_out.txt")
        sim.write_output(out_name=out)
        df = pd.read_csv(out, sep="\t", index_col=0)
        assert df.shape == (20, 4 * 2)

    def test_true_classes_has_all_columns(self, tmp_path):
        sim = simulate(nrows=20, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "sim_out.txt")
        sim.write_output(out_name=out)
        tc = pd.read_csv(str(tmp_path / "sim_out_true_classes.txt"), sep="\t")
        assert {"Circadian", "Linear", "Const"}.issubset(tc.columns)

    def test_true_classes_all_binary(self, tmp_path):
        sim = simulate(nrows=50, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "sim_out.txt")
        sim.write_output(out_name=out)
        tc = pd.read_csv(str(tmp_path / "sim_out_true_classes.txt"), sep="\t")
        for col in ["Circadian", "Linear", "Const"]:
            assert set(tc[col].unique()).issubset({0, 1})

    def test_true_classes_partition_rows(self, tmp_path):
        sim = simulate(nrows=50, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "sim_out.txt")
        sim.write_output(out_name=out)
        tc = pd.read_csv(str(tmp_path / "sim_out_true_classes.txt"), sep="\t")
        row_sums = tc["Circadian"] + tc["Linear"] + tc["Const"]
        assert (row_sums == 1).all()


# ---------------------------------------------------------------------------
# write_proteomics
# ---------------------------------------------------------------------------


class TestWriteProteomics:
    def test_creates_all_three_files(self, tmp_path):
        sim = simulate(nrows=20, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "sim")
        sim.write_proteomics(out_name=out)
        assert os.path.exists(out + "_with_noise.txt")
        assert os.path.exists(out + "_baseline.txt")
        assert os.path.exists(out + "_true_classes.txt")

    def test_with_noise_has_peptide_protein_columns(self, tmp_path):
        sim = simulate(nrows=10, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "sim")
        sim.write_proteomics(out_name=out)
        df = pd.read_csv(out + "_with_noise.txt", sep="\t", index_col=0)
        assert "Protein" in df.columns

    def test_with_noise_null_for_missing(self, tmp_path):
        sim = simulate(nrows=50, tpoints=4, nreps=2, p_miss=0.5, rseed=1)
        out = str(tmp_path / "sim")
        sim.write_proteomics(out_name=out)
        # pandas reads "NULL" back as NaN; confirm missing entries are present
        df = pd.read_csv(out + "_with_noise.txt", sep="\t", index_col=0)
        data_cols = [c for c in df.columns if c.startswith("ZT")]
        assert df[data_cols].isna().any().any()

    def test_true_classes_has_circadian_column(self, tmp_path):
        sim = simulate(nrows=20, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "sim")
        sim.write_proteomics(out_name=out)
        tc = pd.read_csv(out + "_true_classes.txt", sep="\t", index_col=0)
        assert "Circadian" in tc.columns

    def test_true_classes_length(self, tmp_path):
        nrows = 25
        sim = simulate(nrows=nrows, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "sim")
        sim.write_proteomics(out_name=out)
        tc = pd.read_csv(out + "_true_classes.txt", sep="\t", index_col=0)
        assert len(tc) == nrows
