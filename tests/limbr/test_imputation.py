import numpy as np
import pandas as pd
import pytest

from circ.limbr.imputation import imputable


def test_init_loads_data(imputation_file):
    obj = imputable(imputation_file, missingness=0.5)
    assert obj.data is not None
    assert obj.miss == 0.5
    assert obj.pats == {}


def test_init_custom_neighbors(imputation_file):
    obj = imputable(imputation_file, missingness=0.5, neighbors=7)
    assert obj.NN == 7


def test_deduplicate_creates_multiindex(imputation_file):
    obj = imputable(imputation_file, missingness=0.5)
    obj.deduplicate()
    assert isinstance(obj.data.index, pd.MultiIndex)
    assert obj.data.index.names == ["Peptide", "Protein"]


def test_deduplicate_no_duplicate_peptides(imputation_file):
    obj = imputable(imputation_file, missingness=0.5)
    obj.deduplicate()
    counts = obj.data.groupby(level="Peptide").size()
    assert (counts == 1).all()


def test_drop_missing_removes_high_missing_rows(imputation_file):
    obj = imputable(imputation_file, missingness=0.0)  # keep only complete rows
    obj.deduplicate()
    obj.drop_missing()
    # only the 50 complete rows should remain
    assert obj.data.isnull().sum().sum() == 0
    assert len(obj.data) == 50


def test_drop_missing_high_threshold_keeps_all(imputation_file):
    obj = imputable(imputation_file, missingness=1.0)
    obj.deduplicate()
    n_before = len(obj.data)
    obj.drop_missing()
    assert len(obj.data) == n_before


def test_impute_fills_all_nan(imputation_file, tmp_path):
    out = str(tmp_path / "imputed.txt")
    obj = imputable(imputation_file, missingness=0.5, neighbors=5)
    obj.deduplicate()
    obj.drop_missing()
    obj.impute(out)
    result = pd.read_csv(out, sep="\t", index_col=[0, 1])
    assert result.isnull().sum().sum() == 0


def test_impute_preserves_complete_row_count(imputation_file, tmp_path):
    out = str(tmp_path / "imputed.txt")
    obj = imputable(imputation_file, missingness=0.5, neighbors=5)
    obj.deduplicate()
    obj.drop_missing()
    n_rows = len(obj.data)
    obj.impute(out)
    result = pd.read_csv(out, sep="\t", index_col=[0, 1])
    assert len(result) == n_rows


def test_impute_data_pipeline(imputation_file, tmp_path):
    out = str(tmp_path / "pipeline.txt")
    obj = imputable(imputation_file, missingness=0.5, neighbors=5)
    obj.impute_data(out)
    result = pd.read_csv(out, sep="\t", index_col=[0, 1])
    assert result.isnull().sum().sum() == 0
    assert len(result) > 0


def test_impute_writes_parquet(imputation_file, tmp_path):
    out = str(tmp_path / "imputed.parquet")
    obj = imputable(imputation_file, missingness=0.5, neighbors=5)
    obj.impute_data(out)
    result = pd.read_parquet(out)
    assert result.isnull().sum().sum() == 0
    assert len(result) > 0


def test_init_accepts_dataframe(imputation_file):
    df = pd.read_csv(imputation_file, sep="\t")
    obj = imputable(df, missingness=0.5)
    assert obj.data is not None
    assert list(obj.data.columns[:2]) == ["Peptide", "Protein"]


def test_init_accepts_multiindex_dataframe(imputation_file):
    df = pd.read_csv(imputation_file, sep="\t").set_index(["Peptide", "Protein"])
    obj = imputable(df, missingness=0.5)
    # MultiIndex should have been reset to flat columns for deduplicate()
    assert "Peptide" in obj.data.columns
    assert "Protein" in obj.data.columns


# ---------------------------------------------------------------------------
# Regression tests: KNN neighbor handling (#39, #40)
# ---------------------------------------------------------------------------


def test_impute_keeps_nearest_neighbor(tmp_path):
    # #39: query q matches a exactly on observed columns (a is the nearest
    # neighbor) and b is far. With neighbors=2 the imputed value should be
    # mean(a, b) = 50, not 0 (which is what dropping the nearest neighbor gave).
    df = pd.DataFrame(
        [[100.0, 5.0, 5.0], [0.0, 50.0, 50.0], [np.nan, 5.0, 5.0]],
        columns=["ZT00_1", "ZT06_1", "ZT12_1"],
    )
    df.insert(0, "Protein", ["pA", "pB", "pQ"])
    df.insert(0, "Peptide", ["a", "b", "q"])
    out = str(tmp_path / "o.txt")
    imputable(df, missingness=0.5, neighbors=2).impute_data(out)
    result = pd.read_csv(out, sep="\t")
    imputed = result.query("Peptide == 'q'")["ZT00_1"].iloc[0]
    assert imputed == pytest.approx(50.0)


def test_impute_clamps_neighbors_to_complete_cases(tmp_path):
    # #40: fewer complete-case rows than requested neighbors must not raise.
    cols = [f"ZT{2 * i:02d}_1" for i in range(6)]
    rows = [list(np.random.RandomState(i).rand(6)) for i in range(4)]  # 4 complete
    for i in range(4, 6):
        r = list(np.random.RandomState(i).rand(6))
        r[0] = np.nan
        rows.append(r)  # 2 incomplete
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "Protein", [f"p{i}" for i in range(6)])
    df.insert(0, "Peptide", [f"pep{i}" for i in range(6)])
    out = str(tmp_path / "o.txt")
    imputable(df, missingness=0.5, neighbors=10).impute_data(out)  # must not raise
    assert pd.read_csv(out, sep="\t").shape[0] == 6


def test_impute_raises_without_complete_cases(tmp_path):
    # Guard: no complete-case row to learn from -> clear error, not a crash.
    cols = ["ZT00_1", "ZT06_1"]
    df = pd.DataFrame([[np.nan, 1.0], [2.0, np.nan]], columns=cols)
    df.insert(0, "Protein", ["pA", "pB"])
    df.insert(0, "Peptide", ["a", "b"])
    out = str(tmp_path / "o.txt")
    with pytest.raises(ValueError, match="complete-case"):
        imputable(df, missingness=0.9, neighbors=2).impute_data(out)
