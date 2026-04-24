"""Tests for pure-Python utility functions in bin/BooteJTK.py."""
import os
import math
import tempfile

import numpy as np
import pytest

import circ.bootjtk.BooteJTK as B


class TestIsNumber:
    def test_integer_string(self):
        assert B.is_number('42')

    def test_float_string(self):
        assert B.is_number('3.14')

    def test_negative(self):
        assert B.is_number('-1.5')

    def test_na(self):
        assert not B.is_number('NA')

    def test_empty(self):
        assert not B.is_number('')

    def test_inf_is_false(self):
        assert not B.is_number('inf')

    def test_nan_is_false(self):
        assert not B.is_number('nan')


class TestSeriesChar:
    SERIES = ['gene1', '1.0', '3.0', '2.0', '5.0', '4.0']

    def test_max(self):
        mmax, _, _ = B.series_char(self.SERIES)
        assert mmax == pytest.approx(5.0)

    def test_min(self):
        _, mmin, _ = B.series_char(self.SERIES)
        assert mmin == pytest.approx(1.0)

    def test_diff(self):
        _, _, diff = B.series_char(self.SERIES)
        assert diff == pytest.approx(4.0)

    def test_all_same(self):
        series = ['g', '3.0', '3.0', '3.0']
        mmax, mmin, diff = B.series_char(series)
        assert mmax == pytest.approx(3.0)
        assert diff == pytest.approx(0.0)

    def test_with_na(self):
        series = ['g', 'NA', '2.0', '4.0']
        mmax, mmin, diff = B.series_char(series)
        assert mmax == pytest.approx(4.0)
        assert mmin == pytest.approx(2.0)


class TestSeriesMean:
    def test_simple_mean(self):
        series = ['gene', '1.0', '2.0', '3.0', '4.0', '5.0']
        assert B.series_mean(series) == pytest.approx(3.0)

    def test_ignores_na(self):
        series = ['gene', '1.0', 'NA', '3.0']
        assert B.series_mean(series) == pytest.approx(2.0)

    def test_single_value(self):
        series = ['gene', '7.5']
        assert B.series_mean(series) == pytest.approx(7.5)


class TestSeriesStd:
    def test_known_std(self):
        series = ['gene', '2.0', '4.0', '4.0', '4.0', '5.0', '5.0', '7.0', '9.0']
        result = B.series_std(series)
        expected = np.std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert result == pytest.approx(expected, rel=1e-5)

    def test_constant_series_zero_std(self):
        series = ['gene', '3.0', '3.0', '3.0']
        assert B.series_std(series) == pytest.approx(0.0)


class TestFC:
    def test_simple_ratio(self):
        series = ['gene', '1.0', '2.0', '4.0']
        assert B.FC(series) == pytest.approx(4.0)

    def test_all_same(self):
        series = ['gene', '5.0', '5.0', '5.0']
        assert B.FC(series) == pytest.approx(1.0)

    def test_zero_min_returns_sentinel(self):
        series = ['gene', '0.0', '5.0', '10.0']
        result = B.FC(series)
        assert result == -10000


class TestIQRFC:
    # IQR_FC requires at least 5 numeric values (checked inside __score_at_percentile__)
    def test_known_iqr_fc(self):
        series = ['gene', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0']
        result = B.IQR_FC(series)
        assert isinstance(result, float)
        assert result > 0

    def test_returns_nan_for_short_series(self):
        series = ['gene', '1.0', '2.0', '3.0']
        result = B.IQR_FC(series)
        assert result == 'NA' or (isinstance(result, float) and math.isnan(result))

    def test_returns_nan_when_q1_is_zero(self):
        # If Q1 == 0 IQR_FC should return nan
        series = ['gene', '0.0', '0.0', '0.0', '5.0', '10.0', '15.0']
        result = B.IQR_FC(series)
        assert result == 'NA' or result == 0 or (isinstance(result, float) and math.isnan(result))


class TestReadIn:
    def test_hash_header(self, tmp_path):
        f = tmp_path / 'data.txt'
        f.write_text('#\tZT0\tZT2\tZT4\ngene1\t1.0\t2.0\t3.0\n')
        header, data = B.read_in(str(f))
        assert header == ['ZT0', 'ZT2', 'ZT4']
        assert data == [['gene1', '1.0', '2.0', '3.0']]

    def test_id_header(self, tmp_path):
        f = tmp_path / 'data.txt'
        f.write_text('ID\tZT0\tZT2\ngeneA\t5.0\t6.0\ngeneB\t7.0\t8.0\n')
        header, data = B.read_in(str(f))
        assert header == ['ZT0', 'ZT2']
        assert len(data) == 2
        assert data[0][0] == 'geneA'

    def test_multiple_genes(self, tmp_path):
        f = tmp_path / 'data.txt'
        lines = '#\t' + '\t'.join(f'ZT{i}' for i in range(0, 24, 2)) + '\n'
        lines += 'g1\t' + '\t'.join(['1.0'] * 12) + '\n'
        lines += 'g2\t' + '\t'.join(['2.0'] * 12) + '\n'
        f.write_text(lines)
        header, data = B.read_in(str(f))
        assert len(header) == 12
        assert len(data) == 2


class TestDictData:
    def test_keys_are_gene_ids(self):
        data = [['gene1', '1.0', '2.0'], ['gene2', '3.0', '4.0']]
        d = B.dict_data(data)
        assert set(d.keys()) == {'gene1', 'gene2'}

    def test_values_are_full_rows(self):
        data = [['gene1', '1.0', '2.0']]
        d = B.dict_data(data)
        assert d['gene1'] == ['gene1', '1.0', '2.0']


class TestGetData2:
    def test_structure(self):
        header = ['ZT0', 'ZT2', 'ZT4', 'ZT6']
        data = [['g1', '1.0', '2.0', '3.0', '4.0']]
        d_data, new_h = B.get_data2(header, data, 24.0)
        assert 'g1' in d_data
        means, stds, ns = d_data['g1']
        assert len(means) == 4
        assert len(stds) == 4
        assert len(ns) == 4

    def test_single_replicate_std_is_zero(self):
        # Each timepoint appears once → std=0
        header = ['ZT0', 'ZT2', 'ZT4']
        data = [['g1', '1.0', '2.0', '3.0']]
        d_data, _ = B.get_data2(header, data, 24.0)
        means, stds, ns = d_data['g1']
        assert all(s == pytest.approx(0.0) for s in stds)

    def test_two_replicates_per_timepoint(self):
        # get_data2 preserves the full header length (one entry per position,
        # not per unique timepoint); replicated timepoints get the same mean
        header = ['ZT0', 'ZT0', 'ZT2', 'ZT2']
        data = [['g1', '1.0', '3.0', '2.0', '4.0']]
        d_data, new_h = B.get_data2(header, data, 24.0)
        means, stds, ns = d_data['g1']
        # Length == len(header), repeated timepoints share the same mean
        assert len(means) == len(header)
        assert means[0] == pytest.approx(means[1])   # both ZT0 slots same mean
        assert means[0] == pytest.approx(2.0)        # mean of 1.0 and 3.0

    def test_new_header_returns_same_length_as_input(self):
        header = ['ZT0', 'ZT0', 'ZT2', 'ZT2', 'ZT4']
        data = [['g1', '1.0', '2.0', '3.0', '4.0', '5.0']]
        _, new_h = B.get_data2(header, data, 24.0)
        assert len(new_h) == len(header)


class TestDictOrderProbs:
    def test_returns_dict_and_array(self):
        np.random.seed(0)
        d, s3 = B.dict_order_probs([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], [2, 2, 2], size=20)
        assert isinstance(d, dict)
        assert isinstance(s3, np.ndarray)

    def test_probabilities_sum_to_one(self):
        np.random.seed(1)
        d, _ = B.dict_order_probs([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], [2, 2, 2], size=100)
        assert sum(d.values()) == pytest.approx(1.0, abs=1e-6)

    def test_bootstrap_array_shape(self):
        size = 50
        np.random.seed(2)
        _, s3 = B.dict_order_probs([0.0, 1.0, 2.0, 3.0], [0.5]*4, [1]*4, size=size)
        assert s3.shape == (size, 4)

    def test_rank_tuples_correct_length(self):
        np.random.seed(3)
        d, _ = B.dict_order_probs([0.0, 1.0, 2.0], [0.1]*3, [1]*3, size=20)
        for key in d:
            assert len(key) == 3


class TestGenerateModSeries:
    def test_perfect_agreement_tau_one(self):
        x = list(range(1, 9))
        tau, p = B.generate_mod_series(x, x)
        assert tau == pytest.approx(1.0)

    def test_perfect_disagreement_tau_minus_one(self):
        x = list(range(1, 9))
        y = list(reversed(x))
        tau, p = B.generate_mod_series(x, y)
        assert tau == pytest.approx(-1.0)

    def test_p_value_is_one_tailed(self):
        # p should be in (0, 0.5] for strong agreement (tau near 1)
        x = list(range(1, 13))
        tau, p = B.generate_mod_series(x, x)
        assert tau == pytest.approx(1.0)
        assert 0.0 < p <= 0.5  # one-tailed → max 0.5


class TestReadExampleFile:
    """Smoke test against the actual example data."""
    EXAMPLE = os.path.join(os.path.dirname(__file__), '..', 'example', 'TestInput4.txt')

    def test_reads_header(self):
        header, data = B.read_in(self.EXAMPLE)
        assert len(header) == 24
        assert header[0] == 'ZT0'

    def test_reads_multiple_genes(self):
        _, data = B.read_in(self.EXAMPLE)
        assert len(data) > 1

    def test_get_data2_on_example(self):
        header, data = B.read_in(self.EXAMPLE)
        d_data, new_h = B.get_data2(header, data[:5], 24.0)
        assert len(d_data) == 5
        # new_h preserves the full header length (24), not unique timepoints
        assert len(new_h) == len(header)


class TestRefDir:
    def test_ref_dir_attribute_exists(self):
        assert hasattr(B, '_REF_DIR')

    def test_ref_dir_is_a_directory(self):
        assert os.path.isdir(B._REF_DIR)

    def test_period_file_exists(self):
        assert os.path.isfile(os.path.join(B._REF_DIR, 'period24.txt'))

    def test_phases_file_exists(self):
        assert os.path.isfile(os.path.join(B._REF_DIR, 'phases_00-22_by2.txt'))

    def test_asymmetries_file_exists(self):
        assert os.path.isfile(os.path.join(B._REF_DIR, 'asymmetries_02-22_by2.txt'))


class TestProcessGene:
    """Tests for the _process_gene worker function."""

    EXAMPLE = os.path.join(os.path.dirname(__file__), '..', 'example', 'TestInput4.txt')

    @pytest.fixture(autouse=True)
    def setup(self):
        from circ.bootjtk.get_stat_probs import get_waveform_list, make_references, rank_references
        header, data = B.read_in(self.EXAMPLE)
        self.header = header
        d_data, new_header = B.get_data2(header, data[:1], 24.0)
        self.gene_id = list(d_data.keys())[0]
        self.d_data_item = d_data[self.gene_id]

        phases = np.array(list(range(0, 24, 2)), dtype=float)
        widths = np.array(list(range(2, 24, 2)), dtype=float)
        periods = np.array([24.0])
        triples = get_waveform_list(periods, phases, widths)
        dref = make_references(new_header, triples)
        ref_ranks = rank_references(dref, triples)
        B._init_worker(triples, dref, new_header, ref_ranks)

    def test_returns_tuple_with_gene_id(self):
        args = (self.gene_id, self.d_data_item, None, None, 10, 'DEFAULT', 'cosine')
        result = B._process_gene(args)
        assert result is not None
        assert result[0] == self.gene_id

    def test_output_line_is_string_list(self):
        args = (self.gene_id, self.d_data_item, None, None, 10, 'DEFAULT', 'cosine')
        result = B._process_gene(args)
        assert result is not None
        _, out_line, *_ = result
        assert isinstance(out_line, list)
        assert all(isinstance(v, str) for v in out_line)

    def test_output_has_expected_fields(self):
        args = (self.gene_id, self.d_data_item, None, None, 10, 'DEFAULT', 'cosine')
        result = B._process_gene(args)
        assert result is not None
        _, out_line, *_ = result
        # First two fields are gene ID and waveform name
        assert out_line[0] == self.gene_id
        assert out_line[1] == 'cosine'
