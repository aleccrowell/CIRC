import pytest
import numpy as np
import pandas as pd
import os

from circ.pirs.simulations import simulate, analyze


class TestSimulateInit:
    def test_default_instantiation(self):
        sim = simulate()
        assert sim is not None

    def test_sim_array_shape(self):
        sim = simulate(tpoints=4, nrows=20, nreps=2, tpoint_space=2)
        assert sim.sim.shape == (20, 4 * 2)

    def test_column_count(self):
        sim = simulate(tpoints=6, nrows=10, nreps=3, tpoint_space=2)
        assert len(sim.cols) == 6 * 3

    def test_column_name_format(self):
        sim = simulate(tpoints=4, nrows=10, nreps=2, tpoint_space=2)
        assert sim.cols[0] == "ZT02_1"
        assert sim.cols[1] == "ZT02_2"
        assert sim.cols[2] == "ZT04_1"

    def test_const_array_length(self):
        sim = simulate(nrows=50)
        assert len(sim.const) == 50

    def test_const_array_binary(self):
        sim = simulate(nrows=100)
        assert set(np.unique(sim.const)).issubset({0, 1})

    def test_pcirc_controls_circadian_fraction(self):
        sim_high = simulate(nrows=10000, pcirc=0.8, plin=0.1, rseed=1)
        sim_low = simulate(nrows=10000, pcirc=0.1, plin=0.1, rseed=1)
        assert np.mean(sim_high.const) < np.mean(sim_low.const)

    def test_reproducibility_with_same_seed(self):
        sim1 = simulate(nrows=50, rseed=42)
        sim2 = simulate(nrows=50, rseed=42)
        np.testing.assert_array_equal(sim1.sim, sim2.sim)
        np.testing.assert_array_equal(sim1.const, sim2.const)

    def test_different_seeds_give_different_results(self):
        sim1 = simulate(nrows=50, rseed=0)
        sim2 = simulate(nrows=50, rseed=99)
        assert not np.array_equal(sim1.sim, sim2.sim)

    def test_custom_tpoint_space(self):
        sim = simulate(tpoints=4, nrows=10, nreps=2, tpoint_space=4)
        assert sim.cols[0] == "ZT04_1"
        assert sim.cols[2] == "ZT08_1"

    def test_sim_values_scaled(self):
        sim = simulate(nrows=50, rseed=7)
        stds = np.std(sim.sim, axis=1)
        np.testing.assert_allclose(stds, np.ones(50), atol=1e-10)


class TestSimulateWriteOutput:
    def test_write_output_creates_data_file(self, tmp_path):
        sim = simulate(nrows=20, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "sim_out.txt")
        sim.write_output(out_name=out)
        assert os.path.exists(out)

    def test_write_output_creates_classes_file(self, tmp_path):
        sim = simulate(nrows=20, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "sim_out.txt")
        sim.write_output(out_name=out)
        classes_path = str(tmp_path / "sim_out_true_classes.txt")
        assert os.path.exists(classes_path)

    def test_write_output_data_shape(self, tmp_path):
        sim = simulate(nrows=20, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "sim_out.txt")
        sim.write_output(out_name=out)
        df = pd.read_csv(out, sep="\t", index_col=0)
        assert df.shape == (20, 4 * 2)

    def test_write_output_classes_length(self, tmp_path):
        sim = simulate(nrows=20, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "sim_out.txt")
        sim.write_output(out_name=out)
        classes = pd.read_csv(str(tmp_path / "sim_out_true_classes.txt"), sep="\t")
        assert len(classes) == 20

    def test_write_output_classes_column(self, tmp_path):
        sim = simulate(nrows=20, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "sim_out.txt")
        sim.write_output(out_name=out)
        classes = pd.read_csv(str(tmp_path / "sim_out_true_classes.txt"), sep="\t")
        assert "Const" in classes.columns


class TestSimulateWriteGenorm:
    def test_creates_file(self, tmp_path):
        sim = simulate(nrows=10, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "genorm_out.txt")
        sim.write_genorm(out_name=out)
        assert os.path.exists(out)

    def test_output_has_required_columns(self, tmp_path):
        sim = simulate(nrows=10, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "genorm_out.txt")
        sim.write_genorm(out_name=out)
        df = pd.read_csv(out, sep=" ")
        assert set(["Sample", "Detector", "Cq"]).issubset(df.columns)

    def test_cq_normalized_range(self, tmp_path):
        sim = simulate(nrows=10, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "genorm_out.txt")
        sim.write_genorm(out_name=out)
        df = pd.read_csv(out, sep=" ")
        assert df["Cq"].min() >= 0.0
        assert df["Cq"].max() <= 1.0

    def test_sample_prefix(self, tmp_path):
        sim = simulate(nrows=10, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "genorm_out.txt")
        sim.write_genorm(out_name=out)
        df = pd.read_csv(out, sep=" ")
        assert all(df["Sample"].str.startswith("timepoint_"))

    def test_detector_prefix(self, tmp_path):
        sim = simulate(nrows=10, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "genorm_out.txt")
        sim.write_genorm(out_name=out)
        df = pd.read_csv(out, sep=" ")
        assert all(df["Detector"].str.startswith("gene_"))


class TestSimulateWriteNormfinder:
    def test_creates_file(self, tmp_path):
        sim = simulate(nrows=10, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "normfinder_out.txt")
        sim.write_normfinder(out_name=out)
        assert os.path.exists(out)

    def test_output_has_required_columns(self, tmp_path):
        sim = simulate(nrows=10, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "normfinder_out.txt")
        sim.write_normfinder(out_name=out)
        df = pd.read_csv(out, sep=" ")
        assert set(["Sample", "Detector", "Cq"]).issubset(df.columns)

    def test_cq_normalized_range(self, tmp_path):
        sim = simulate(nrows=10, tpoints=4, nreps=2, rseed=0)
        out = str(tmp_path / "normfinder_out.txt")
        sim.write_normfinder(out_name=out)
        df = pd.read_csv(out, sep=" ")
        assert df["Cq"].min() >= 0.0
        assert df["Cq"].max() <= 1.0


class TestAnalyze:
    def _make_output_files(self, tmp_path, sim, tag, rep=0):
        """Helper: write sim output and a PIRS scores file for analyze."""
        data_out = str(tmp_path / f"sim_{rep}.txt")
        sim.write_output(out_name=data_out)

        from circ.pirs.rank import ranker

        r = ranker(data_out, anova=False)
        r.get_tpoints()
        scores = r.calculate_scores()
        scores_path = str(tmp_path / f"scores_{tag}_{rep}.txt")
        scores.to_csv(scores_path, sep="\t")
        classes_path = data_out[:-4] + "_true_classes.txt"
        return scores_path, classes_path

    def test_add_classes(self, tmp_path):
        sim = simulate(nrows=30, tpoints=4, nreps=2, rseed=1)
        data_out = str(tmp_path / "sim.txt")
        sim.write_output(out_name=data_out)
        classes_path = data_out[:-4] + "_true_classes.txt"

        a = analyze()
        a.add_classes(classes_path, rep=0)
        assert len(a.true_classes) == 30

    def test_add_classes_rep_column(self, tmp_path):
        sim = simulate(nrows=20, tpoints=4, nreps=2, rseed=2)
        data_out = str(tmp_path / "sim.txt")
        sim.write_output(out_name=data_out)
        classes_path = data_out[:-4] + "_true_classes.txt"

        a = analyze()
        a.add_classes(classes_path, rep=5)
        assert (a.true_classes["rep"] == 5).all()

    def test_add_classes_accumulates_multiple_reps(self, tmp_path):
        a = analyze()
        for rep in range(3):
            sim = simulate(nrows=10, tpoints=4, nreps=2, rseed=rep)
            out = str(tmp_path / f"sim_{rep}.txt")
            sim.write_output(out_name=out)
            a.add_classes(out[:-4] + "_true_classes.txt", rep=rep)
        assert len(a.true_classes) == 30

    def test_add_data(self, tmp_path):
        sim = simulate(nrows=30, tpoints=4, nreps=2, rseed=3)
        scores_path, _ = self._make_output_files(tmp_path, sim, "pirs", rep=0)

        a = analyze()
        a.add_data(scores_path, tag="pirs", rep=0)
        assert not a.merged.empty
        assert (a.merged["method"] == "pirs").all()

    def test_add_data_accumulates_methods(self, tmp_path):
        sim = simulate(nrows=30, tpoints=4, nreps=2, rseed=4)
        scores_path, _ = self._make_output_files(tmp_path, sim, "pirs", rep=0)

        a = analyze()
        a.add_data(scores_path, tag="pirs", rep=0)
        a.add_data(scores_path, tag="rsd", rep=0)
        assert set(a.merged["method"].unique()) == {"pirs", "rsd"}
