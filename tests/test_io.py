"""Tests for circ.io — shared expression I/O utilities."""
import pandas as pd
import pytest

from circ.io import read_expression, sidecar_path, write_expression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def expr_df():
    """Minimal RNAseq-style expression DataFrame."""
    df = pd.DataFrame(
        {"ZT02_1": [1.0, 2.0], "ZT04_1": [1.5, 2.5]},
        index=["g1", "g2"],
    )
    df.index.name = "#"
    return df


@pytest.fixture
def prot_df():
    """Minimal proteomics-style expression DataFrame with MultiIndex."""
    idx = pd.MultiIndex.from_tuples([("pep_a", "prot_1"), ("pep_b", "prot_1")],
                                    names=["Peptide", "Protein"])
    return pd.DataFrame({"ZT02_1": [1.0, 2.0], "ZT04_1": [1.5, 2.5]}, index=idx)


# ---------------------------------------------------------------------------
# read_expression
# ---------------------------------------------------------------------------

class TestReadExpression:
    def test_passthrough_dataframe(self, expr_df):
        result = read_expression(expr_df)
        pd.testing.assert_frame_equal(result, expr_df)

    def test_passthrough_returns_copy(self, expr_df):
        result = read_expression(expr_df)
        result.iloc[0, 0] = 999.0
        assert expr_df.iloc[0, 0] != 999.0

    def test_read_tsv_rnaseq(self, tmp_path, expr_df):
        path = str(tmp_path / "expr.txt")
        expr_df.to_csv(path, sep="\t")
        result = read_expression(path, data_type="r")
        pd.testing.assert_frame_equal(result, expr_df)

    def test_read_tsv_proteomics(self, tmp_path, prot_df):
        path = str(tmp_path / "prot.txt")
        prot_df.reset_index().to_csv(path, sep="\t", index=False)
        result = read_expression(path, data_type="p")
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["Peptide", "Protein"]

    def test_read_parquet_rnaseq(self, tmp_path, expr_df):
        path = str(tmp_path / "expr.parquet")
        expr_df.to_parquet(path)
        result = read_expression(path)
        pd.testing.assert_frame_equal(result, expr_df)

    def test_read_parquet_proteomics(self, tmp_path, prot_df):
        path = str(tmp_path / "prot.parquet")
        prot_df.to_parquet(path)
        result = read_expression(path, data_type="p")
        pd.testing.assert_frame_equal(result, prot_df)


# ---------------------------------------------------------------------------
# write_expression
# ---------------------------------------------------------------------------

class TestWriteExpression:
    def test_write_tsv(self, tmp_path, expr_df):
        path = str(tmp_path / "out.txt")
        write_expression(expr_df, path)
        result = pd.read_csv(path, sep="\t", index_col=0)
        pd.testing.assert_frame_equal(result, expr_df)

    def test_write_parquet(self, tmp_path, expr_df):
        path = str(tmp_path / "out.parquet")
        write_expression(expr_df, path)
        result = pd.read_parquet(path)
        pd.testing.assert_frame_equal(result, expr_df)

    def test_write_parquet_multiindex(self, tmp_path, prot_df):
        path = str(tmp_path / "prot.parquet")
        write_expression(prot_df, path)
        result = pd.read_parquet(path)
        pd.testing.assert_frame_equal(result, prot_df)

    def test_roundtrip_parquet(self, tmp_path, expr_df):
        path = str(tmp_path / "roundtrip.parquet")
        write_expression(expr_df, path)
        result = read_expression(path)
        pd.testing.assert_frame_equal(result, expr_df)


# ---------------------------------------------------------------------------
# sidecar_path
# ---------------------------------------------------------------------------

class TestSidecarPath:
    def test_preserves_parquet_extension(self):
        assert sidecar_path("/data/out.parquet", "_trends") == "/data/out_trends.parquet"

    def test_preserves_txt_extension(self):
        assert sidecar_path("/data/out.txt", "_perms") == "/data/out_perms.txt"

    def test_preserves_tsv_extension(self):
        assert sidecar_path("/data/result.tsv", "_tks") == "/data/result_tks.tsv"

    def test_no_directory_component(self):
        assert sidecar_path("out.parquet", "_bias") == "out_bias.parquet"
