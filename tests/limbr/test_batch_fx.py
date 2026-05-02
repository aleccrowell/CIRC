import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from circ.limbr.batch_fx import sva


def test_init_rnaseq(rnaseq_file):
    obj = sva(rnaseq_file, design="c", data_type="r")
    assert obj.data_type == "r"
    assert obj.designtype == "c"
    assert obj.raw_data is not None
    assert obj.notdone is True
    assert obj.norm_map is None


def test_init_proteomics(proteomics_file):
    obj = sva(proteomics_file, design="c", data_type="p")
    assert obj.data_type == "p"
    assert isinstance(obj.raw_data.index, pd.MultiIndex)


def test_init_block_design(rnaseq_file, block_design_file):
    obj = sva(rnaseq_file, design="b", data_type="r", blocks=block_design_file)
    assert hasattr(obj, "block_design")
    assert isinstance(obj.block_design, list)
    assert len(obj.block_design) == obj.raw_data.shape[1]


def test_pool_normalize_preserves_shape(rnaseq_file):
    obj = sva(rnaseq_file, design="c", data_type="r")
    obj.pool_normalize()
    assert obj.data.shape == obj.raw_data.shape


def test_pool_normalize_sets_scaler(rnaseq_file):
    obj = sva(rnaseq_file, design="c", data_type="r")
    obj.pool_normalize()
    assert isinstance(obj.scaler, StandardScaler)


def test_pool_normalize_preserves_index(rnaseq_file):
    obj = sva(rnaseq_file, design="c", data_type="r")
    obj.pool_normalize()
    assert set(obj.data.index) == set(obj.raw_data.index)


def test_get_tpoints_length(rnaseq_file):
    obj = sva(rnaseq_file, design="c", data_type="r")
    obj.pool_normalize()
    obj.get_tpoints()
    assert len(obj.tpoints) == obj.data.shape[1]


def test_get_tpoints_values(rnaseq_file):
    obj = sva(rnaseq_file, design="c", data_type="r")
    obj.pool_normalize()
    obj.get_tpoints()
    assert obj.tpoints[0] == 2
    assert obj.tpoints[1] == 2
    assert obj.tpoints[2] == 4


def test_prim_cor_circadian_length(rnaseq_file):
    obj = sva(rnaseq_file, design="c", data_type="r")
    obj.pool_normalize()
    obj.get_tpoints()
    obj.prim_cor()
    assert len(obj.cors) == obj.data.shape[0]


def test_prim_cor_4h_sampling_nonzero(tmp_path):
    """circ_cor with 4h sampling must not alias all correlations to zero.

    The old hardcoded per=12 aliased to zero with only 6 unique timepoints
    (12 > 6, so np.roll wrapped to a no-op shift for both per and per//2).
    The fix derives per=6 from the 4h spacing, giving a valid half-period shift.
    """
    cols = [f"ZT{tp:02d}_{r}" for tp in range(4, 25, 4) for r in (1, 2)]
    np.random.seed(0)
    df = pd.DataFrame(np.random.randn(30, len(cols)), columns=cols)
    df.index = [f"gene_{i}" for i in range(30)]
    df.index.name = "#"
    path = str(tmp_path / "rnaseq_4h.txt")
    df.to_csv(path, sep="\t")
    obj = sva(path, design="c", data_type="r")
    obj.pool_normalize()
    obj.get_tpoints()
    obj.prim_cor()
    assert not np.all(obj.cors == 0), (
        "all cors zero — per likely aliased to a no-op shift"
    )


def test_prim_cor_timecourse_length(rnaseq_file):
    obj = sva(rnaseq_file, design="t", data_type="r")
    obj.pool_normalize()
    obj.get_tpoints()
    obj.prim_cor()
    assert len(obj.cors) == obj.data.shape[0]


def test_prim_cor_block_length(rnaseq_file, block_design_file):
    obj = sva(rnaseq_file, design="b", data_type="r", blocks=block_design_file)
    obj.pool_normalize()
    obj.prim_cor()
    assert len(obj.cors) == obj.data.shape[0]


def test_reduce_fewer_rows(rnaseq_file):
    obj = sva(rnaseq_file, design="c", data_type="r")
    obj.pool_normalize()
    obj.get_tpoints()
    obj.prim_cor()
    obj.reduce(perc_red=25)
    assert len(obj.data_reduced) < len(obj.data)


def test_reduce_approximate_percentage(rnaseq_file):
    obj = sva(rnaseq_file, design="c", data_type="r")
    obj.pool_normalize()
    obj.get_tpoints()
    obj.prim_cor()
    obj.reduce(perc_red=50)
    expected = int(len(obj.data) * 0.5)
    assert abs(len(obj.data_reduced) - expected) <= 5


def test_set_res_shape(rnaseq_file):
    obj = sva(rnaseq_file, design="c", data_type="r")
    obj.pool_normalize()
    obj.get_tpoints()
    obj.prim_cor()
    obj.reduce()
    obj.set_res()
    assert obj.res.shape == obj.data_reduced.shape


def test_set_tks_is_variance_ratios(rnaseq_file):
    obj = sva(rnaseq_file, design="c", data_type="r")
    obj.pool_normalize()
    obj.get_tpoints()
    obj.prim_cor()
    obj.reduce()
    obj.set_res()
    obj.set_tks()
    assert isinstance(obj.tks, np.ndarray)
    assert (obj.tks >= 0).all()
    assert obj.tks.sum() <= 1.0 + 1e-6


def test_perm_test_sigs_in_range(rnaseq_file):
    obj = sva(rnaseq_file, design="c", data_type="r")
    obj.preprocess_default()
    obj.perm_test(nperm=5)
    assert len(obj.sigs) == len(obj.tks)
    assert all(0.0 <= s <= 1.0 for s in obj.sigs)


def test_preprocess_default_sets_expected_attrs(rnaseq_file):
    obj = sva(rnaseq_file, design="c", data_type="r")
    obj.preprocess_default()
    assert hasattr(obj, "data")
    assert hasattr(obj, "tpoints")
    assert hasattr(obj, "cors")
    assert hasattr(obj, "data_reduced")
    assert hasattr(obj, "res")
    assert hasattr(obj, "tks")


def test_perm_test_parallel(rnaseq_file):
    obj = sva(rnaseq_file, design="c", data_type="r")
    obj.preprocess_default()
    obj.perm_test(nperm=10, npr=2)
    assert len(obj.sigs) == len(obj.tks)
    assert all(0.0 <= s <= 1.0 for s in obj.sigs)


def test_init_accepts_dataframe(rnaseq_file):
    df = pd.read_csv(rnaseq_file, sep="\t", index_col=0)
    obj = sva(df, design="c", data_type="r")
    assert obj.raw_data is not None
    assert isinstance(obj.raw_data, pd.DataFrame)
    pd.testing.assert_frame_equal(obj.raw_data, df)


def test_init_accepts_parquet(tmp_path, rnaseq_file):
    df = pd.read_csv(rnaseq_file, sep="\t", index_col=0)
    path = str(tmp_path / "rnaseq.parquet")
    df.to_parquet(path)
    obj = sva(path, design="c", data_type="r")
    assert isinstance(obj.raw_data, pd.DataFrame)
    pd.testing.assert_frame_equal(obj.raw_data, df)
