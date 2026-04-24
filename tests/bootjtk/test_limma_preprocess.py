"""Tests for bootjtk/limma_preprocess.py — Python data-plumbing functions."""
import numpy as np
import pandas as pd
import pytest

from circ.bootjtk.limma_preprocess import (
    read_timeseries,
    parse_timepoint_label,
    deduplicate_timepoints,
    deduplicate_rownames,
    prepare_timeseries,
    write_limma_outputs,
)


# ---------------------------------------------------------------------------
# parse_timepoint_label
# ---------------------------------------------------------------------------

class TestParseTimepointLabel:
    def test_zt_prefix(self):
        assert parse_timepoint_label('ZT4') == 4.0

    def test_ct_prefix(self):
        assert parse_timepoint_label('CT12') == 12.0

    def test_x_prefix(self):
        assert parse_timepoint_label('X8') == 8.0

    def test_bare_number(self):
        assert parse_timepoint_label('0') == 0.0

    def test_float_string(self):
        assert parse_timepoint_label('2.5') == 2.5

    def test_leading_whitespace_stripped(self):
        assert parse_timepoint_label(' ZT6') == 6.0

    def test_zt_zero(self):
        assert parse_timepoint_label('ZT0') == 0.0

    def test_large_value(self):
        assert parse_timepoint_label('ZT46') == 46.0


# ---------------------------------------------------------------------------
# deduplicate_timepoints
# ---------------------------------------------------------------------------

class TestDeduplicateTimepoints:
    def test_no_duplicates_unchanged(self):
        times = [0.0, 2.0, 4.0, 6.0]
        assert deduplicate_timepoints(times, 24) == [0.0, 2.0, 4.0, 6.0]

    def test_single_duplicate_offset_by_period(self):
        result = deduplicate_timepoints([0.0, 2.0, 0.0], 24)
        assert result == [0.0, 2.0, 24.0]

    def test_two_duplicates_of_same_value(self):
        result = deduplicate_timepoints([0.0, 0.0, 0.0], 24)
        assert result == [0.0, 24.0, 48.0]

    def test_multiple_different_duplicates(self):
        result = deduplicate_timepoints([0.0, 2.0, 0.0, 4.0, 2.0], 24)
        assert result == [0.0, 2.0, 24.0, 4.0, 26.0]

    def test_result_all_unique(self):
        times = [0.0, 2.0, 4.0, 0.0, 2.0, 4.0, 0.0]
        result = deduplicate_timepoints(times, 24)
        assert len(result) == len(set(result))

    def test_length_preserved(self):
        times = [0.0, 2.0, 0.0, 2.0]
        result = deduplicate_timepoints(times, 24)
        assert len(result) == len(times)

    def test_different_period(self):
        result = deduplicate_timepoints([0.0, 0.0], 12)
        assert result == [0.0, 12.0]

    def test_int_input_converted_to_float(self):
        result = deduplicate_timepoints([0, 2, 4], 24)
        assert all(isinstance(t, float) for t in result)

    def test_real_example_zt0_to_zt46(self):
        # 24 time points ZT0..ZT46 step 2 → no duplicates after stripping prefix
        times = list(range(0, 48, 2))  # 0, 2, 4, ..., 46
        result = deduplicate_timepoints([float(t) for t in times], 24)
        assert len(result) == len(set(result))
        assert result[:12] == [float(t) for t in range(0, 24, 2)]


# ---------------------------------------------------------------------------
# deduplicate_rownames
# ---------------------------------------------------------------------------

class TestDeduplicateRownames:
    def test_no_duplicates_unchanged(self):
        names = ['a', 'b', 'c']
        assert deduplicate_rownames(names) == ['a', 'b', 'c']

    def test_single_duplicate_appends_suffix(self):
        result = deduplicate_rownames(['a', 'b', 'a'])
        assert result == ['a', 'b', 'a-xxx1']

    def test_three_of_same(self):
        result = deduplicate_rownames(['a', 'a', 'a'])
        assert len(set(result)) == 3
        assert result[0] == 'a'

    def test_all_unique_after_dedup(self):
        names = ['gene1', 'gene2', 'gene1', 'gene3', 'gene2']
        result = deduplicate_rownames(names)
        assert len(result) == len(set(result))

    def test_length_preserved(self):
        names = ['a', 'b', 'a', 'c', 'a']
        result = deduplicate_rownames(names)
        assert len(result) == len(names)

    def test_first_occurrence_unchanged(self):
        result = deduplicate_rownames(['x', 'y', 'x'])
        assert result[0] == 'x'
        assert result[1] == 'y'


# ---------------------------------------------------------------------------
# read_timeseries
# ---------------------------------------------------------------------------

class TestReadTimeseries:
    def test_hash_header(self, tmp_path):
        f = tmp_path / 'data.txt'
        f.write_text('#\tZT0\tZT2\tZT4\ngene1\t1.0\t2.0\t3.0\n')
        df, raw_cols = read_timeseries(str(f))
        assert raw_cols == ['ZT0', 'ZT2', 'ZT4']
        assert 'gene1' in df.index

    def test_id_header(self, tmp_path):
        f = tmp_path / 'data.txt'
        f.write_text('ID\tZT0\tZT2\ngeneA\t5.0\t6.0\ngeneB\t7.0\t8.0\n')
        df, raw_cols = read_timeseries(str(f))
        assert raw_cols == ['ZT0', 'ZT2']
        assert list(df.index) == ['geneA', 'geneB']

    def test_na_values_are_nan(self, tmp_path):
        f = tmp_path / 'data.txt'
        f.write_text('#\tZT0\tZT2\ngene1\tNA\t2.0\n')
        df, _ = read_timeseries(str(f))
        assert np.isnan(df.iloc[0, 0])

    def test_values_are_numeric(self, tmp_path):
        f = tmp_path / 'data.txt'
        f.write_text('#\tZT0\tZT2\ngene1\t1.5\t2.5\n')
        df, _ = read_timeseries(str(f))
        assert df.dtypes.iloc[0] == float

    def test_multiple_genes(self, tmp_path):
        lines = '#\t' + '\t'.join(f'ZT{i}' for i in range(0, 24, 2)) + '\n'
        for i in range(5):
            lines += f'gene{i}\t' + '\t'.join(['1.0'] * 12) + '\n'
        f = tmp_path / 'data.txt'
        f.write_text(lines)
        df, raw_cols = read_timeseries(str(f))
        assert len(df) == 5
        assert len(raw_cols) == 12


# ---------------------------------------------------------------------------
# prepare_timeseries
# ---------------------------------------------------------------------------

class TestPrepareTimeseries:
    def _make_file(self, tmp_path, content):
        f = tmp_path / 'data.txt'
        f.write_text(content)
        return str(f)

    def test_column_names_are_float(self, tmp_path):
        fn = self._make_file(tmp_path, '#\tZT0\tZT2\tZT4\ngene1\t1.0\t2.0\t3.0\n')
        df, _ = prepare_timeseries(fn, 24.0)
        assert all(isinstance(c, float) for c in df.columns)

    def test_zt_prefix_stripped(self, tmp_path):
        fn = self._make_file(tmp_path, '#\tZT0\tZT4\ngene1\t1.0\t2.0\n')
        df, _ = prepare_timeseries(fn, 24.0)
        assert list(df.columns) == [0.0, 4.0]

    def test_unique_times_sorted(self, tmp_path):
        fn = self._make_file(tmp_path, '#\tZT4\tZT0\tZT2\ngene1\t1.0\t2.0\t3.0\n')
        _, unique_times = prepare_timeseries(fn, 24.0)
        assert unique_times == sorted(unique_times)

    def test_unique_times_mod_period(self, tmp_path):
        # ZT0..ZT46 step 2 → unique times 0..22 step 2
        cols = '\t'.join(f'ZT{i}' for i in range(0, 48, 2))
        fn = self._make_file(tmp_path, f'#\t{cols}\ngene1\t' + '\t'.join(['1.0'] * 24) + '\n')
        _, unique_times = prepare_timeseries(fn, 24.0)
        assert unique_times == list(range(0, 24, 2))

    def test_duplicate_columns_deduplicated(self, tmp_path):
        fn = self._make_file(tmp_path, '#\tZT0\tZT0\tZT2\ngene1\t1.0\t2.0\t3.0\n')
        df, _ = prepare_timeseries(fn, 24.0)
        assert len(df.columns) == len(set(df.columns))

    def test_duplicate_rows_deduplicated(self, tmp_path):
        fn = self._make_file(tmp_path, '#\tZT0\tZT2\ngene1\t1.0\t2.0\ngene1\t3.0\t4.0\n')
        df, _ = prepare_timeseries(fn, 24.0)
        assert len(df.index) == len(set(df.index))

    def test_index_name_is_id(self, tmp_path):
        fn = self._make_file(tmp_path, '#\tZT0\ngene1\t1.0\n')
        df, _ = prepare_timeseries(fn, 24.0)
        assert df.index.name == 'ID'

    def test_example_file(self):
        import os
        example = os.path.join(
            os.path.dirname(__file__), '..', 'example', 'TestInput4.txt'
        )
        df, unique_times = prepare_timeseries(example, 24.0)
        assert list(unique_times) == list(range(0, 24, 2))
        assert len(df.columns) == 24   # ZT0..ZT46 → 24 unique deduplicated cols
        assert len(df) > 0


# ---------------------------------------------------------------------------
# write_limma_outputs
# ---------------------------------------------------------------------------

class TestWriteLimmaOutputs:
    @pytest.fixture()
    def long_df(self):
        return pd.DataFrame({
            'ID':    ['g1', 'g1', 'g2', 'g2'],
            'Time':  [0.0, 2.0, 0.0, 2.0],
            'Mean':  [1.0, 2.0, 3.0, 4.0],
            'SD':    [0.1, 0.2, 0.3, 0.4],
            'SDpre': [0.5, 0.6, 0.7, 0.8],
            'N':     [2.0, 2.0, 2.0, 2.0],
        })

    def test_means_file_created(self, tmp_path, long_df):
        write_limma_outputs(long_df, str(tmp_path / 'sample'), 'postLimma')
        assert (tmp_path / 'sample_Means_postLimma.txt').exists()

    def test_sds_file_created(self, tmp_path, long_df):
        write_limma_outputs(long_df, str(tmp_path / 'sample'), 'postLimma')
        assert (tmp_path / 'sample_Sds_postLimma.txt').exists()

    def test_ns_file_created(self, tmp_path, long_df):
        write_limma_outputs(long_df, str(tmp_path / 'sample'), 'postLimma')
        assert (tmp_path / 'sample_Ns_postLimma.txt').exists()

    def test_sdspre_file_created(self, tmp_path, long_df):
        write_limma_outputs(long_df, str(tmp_path / 'sample'), 'postLimma')
        assert (tmp_path / 'sample_Sds-pre_postLimma.txt').exists()

    def test_means_values_correct(self, tmp_path, long_df):
        write_limma_outputs(long_df, str(tmp_path / 'sample'), 'postLimma')
        result = pd.read_table(str(tmp_path / 'sample_Means_postLimma.txt'), index_col='ID')
        assert result.loc['g1'].iloc[0] == pytest.approx(1.0)
        assert result.loc['g2'].iloc[1] == pytest.approx(4.0)

    def test_wide_shape(self, tmp_path, long_df):
        write_limma_outputs(long_df, str(tmp_path / 's'), 'postVash')
        means = pd.read_table(str(tmp_path / 's_Means_postVash.txt'), index_col='ID')
        assert means.shape == (2, 2)   # 2 genes × 2 time points

    def test_readable_by_bootjtk_read_in(self, tmp_path, long_df):
        from circ.bootjtk.BooteJTK import read_in
        write_limma_outputs(long_df, str(tmp_path / 's'), 'postLimma')
        header, data = read_in(str(tmp_path / 's_Means_postLimma.txt'))
        assert len(header) == 2       # two time points
        assert len(data) == 2         # two genes
        assert data[0][0] == 'g1'
        assert data[1][0] == 'g2'
