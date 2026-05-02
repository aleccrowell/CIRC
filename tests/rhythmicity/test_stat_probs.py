"""Tests for the get_stat_probs module."""

import math
import numpy as np
import pytest

from circ.rhythmicity.get_stat_probs import (
    kt,
    generate_base_reference,
    get_matches,
    get_stat_probs,
    get_waveform_list,
    make_references,
    rank_references,
    farctanh,
    periodic,
)


class TestKendallTau:
    def test_perfect_agreement(self):
        tau, _ = kt([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert tau == pytest.approx(1.0)

    def test_perfect_disagreement(self):
        tau, _ = kt([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        assert tau == pytest.approx(-1.0)

    def test_known_value(self):
        x = [1, 3, 2, 5, 4, 6, 8, 7]
        y = [2, 1, 4, 3, 5, 7, 6, 8]
        tau, _ = kt(x, y)
        assert tau == pytest.approx(4 / 7, abs=1e-5)  # 4/7 ≈ 0.5714

    def test_returns_two_values(self):
        result = kt([1, 2, 3], [1, 2, 3])
        assert len(result) == 2

    def test_p_value_in_range(self):
        _, p = kt([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
        assert 0.0 <= p <= 1.0

    def test_uncorrelated_tau_near_zero(self):
        # A shuffled sequence should give tau close to 0 on average
        rng = np.random.default_rng(42)
        taus = [kt(list(range(20)), list(rng.permutation(20)))[0] for _ in range(50)]
        assert abs(np.mean(taus)) < 0.3

    def test_with_x_ties(self):
        """Tau-b with ties in x only: tau < 1 even for monotone y."""
        x = [1, 1, 2, 2, 3, 3]
        y = [1, 2, 3, 4, 5, 6]
        tau, _ = kt(x, y)
        assert tau == pytest.approx(0.8944271910, abs=1e-5)

    def test_with_y_ties(self):
        """Tau-b with ties in y only: symmetric result to x-ties case."""
        x = [1, 2, 3, 4, 5, 6]
        y = [1, 1, 2, 2, 3, 3]
        tau, _ = kt(x, y)
        assert tau == pytest.approx(0.8944271910, abs=1e-5)

    def test_with_joint_ties(self):
        """Tau-b with joint (x, y) ties: concordance reduced by tied pairs."""
        x = [1, 1, 2, 3, 4]
        y = [1, 1, 3, 2, 4]
        tau, _ = kt(x, y)
        assert tau == pytest.approx(7 / 9, abs=1e-5)  # 7/9 ≈ 0.7778

    def test_all_tied_returns_nan(self):
        """Completely constant arrays → tot == u == v → (nan, nan)."""
        tau, p = kt([2, 2, 2, 2], [2, 2, 2, 2])
        assert np.isnan(tau)
        assert np.isnan(p)


class TestGenerateBaseReference:
    HEADER = list(range(0, 24, 2))  # 12 evenly-spaced ZT timepoints

    def test_cosine_correct_length(self):
        ref = generate_base_reference(self.HEADER, "cosine", 24.0, 0.0, 12.0)
        assert len(ref) == len(self.HEADER)

    def test_trough_correct_length(self):
        ref = generate_base_reference(self.HEADER, "trough", 24.0, 0.0, 12.0)
        assert len(ref) == len(self.HEADER)

    def test_cosine_peak_at_phase_zero(self):
        # With phase=0 the cosine peak lands at ZT0 (first element)
        ref = generate_base_reference(self.HEADER, "cosine", 24.0, 0.0, 12.0)
        assert ref[0] == pytest.approx(max(ref), abs=1e-6)

    def test_cosine_peak_shifts_with_phase(self):
        ref0 = generate_base_reference(self.HEADER, "cosine", 24.0, 0.0, 12.0)
        ref6 = generate_base_reference(self.HEADER, "cosine", 24.0, 6.0, 12.0)
        # Different phases should produce different waveforms
        assert not np.allclose(ref0, ref6)

    def test_cosine_values_bounded(self):
        ref = generate_base_reference(self.HEADER, "cosine", 24.0, 0.0, 12.0)
        assert all(-1.001 <= v <= 1.001 for v in ref)

    def test_trough_values_bounded(self):
        ref = generate_base_reference(self.HEADER, "trough", 24.0, 0.0, 12.0)
        assert all(-1.001 <= v <= 1.001 for v in ref)

    def test_returns_numpy_array(self):
        ref = generate_base_reference(self.HEADER, "cosine", 24.0, 0.0, 12.0)
        assert isinstance(ref, np.ndarray)

    def test_impulse_correct_length(self):
        ref = generate_base_reference(self.HEADER, "impulse", 24.0, 0.0, 12.0)
        assert len(ref) == len(self.HEADER)

    def test_step_correct_length(self):
        ref = generate_base_reference(self.HEADER, "step", 24.0, 0.0, 12.0)
        assert len(ref) == len(self.HEADER)

    def test_impulse_values_nonnegative(self):
        ref = generate_base_reference(self.HEADER, "impulse", 24.0, 0.0, 12.0)
        assert all(v >= -1e-9 for v in ref)

    def test_step_values_binary(self):
        ref = generate_base_reference(self.HEADER, "step", 24.0, 0.0, 12.0)
        assert all(v in (0.0, 1.0) for v in ref)


class TestGetWaveformList:
    # get_waveform_list pre-allocates int(n_phases * n_widths / 2) slots and
    # deduplicates (phase, nadir) pairs. Use the canonical reference-file inputs
    # (12 phases × 11 widths) so the formula holds.
    PHASES = np.array(list(range(0, 24, 2)))  # 12 values
    WIDTHS = np.array(list(range(2, 24, 2)))  # 11 values

    def test_returns_2d_array(self):
        triples = get_waveform_list(np.array([24]), self.PHASES, self.WIDTHS)
        assert triples.ndim == 2
        assert triples.shape[1] == 3

    def test_positive_row_count(self):
        triples = get_waveform_list(np.array([24]), self.PHASES, self.WIDTHS)
        assert triples.shape[0] > 0

    def test_period_value_present(self):
        triples = get_waveform_list(np.array([24]), self.PHASES, self.WIDTHS)
        assert all(t[0] == 24 for t in triples)

    def test_phase_values_in_input_range(self):
        triples = get_waveform_list(np.array([24]), self.PHASES, self.WIDTHS)
        for t in triples:
            assert t[1] in self.PHASES

    def test_row_count_is_nonzero(self):
        # With the standard 12-phase × 11-width inputs the result is non-empty
        triples = get_waveform_list(np.array([24]), self.PHASES, self.WIDTHS)
        assert triples.shape[0] > 0


class TestMakeReferences:
    HEADER = list(range(0, 24, 2))  # 12 ZT timepoints
    PHASES = np.array(list(range(0, 24, 2)))
    WIDTHS = np.array(list(range(2, 24, 2)))

    def test_returns_dict(self):
        triples = get_waveform_list(np.array([24]), self.PHASES, self.WIDTHS)
        dref = make_references(self.HEADER, triples)
        assert isinstance(dref, dict)

    def test_key_count_matches_triples(self):
        triples = get_waveform_list(np.array([24]), self.PHASES, self.WIDTHS)
        dref = make_references(self.HEADER, triples)
        assert len(dref) == triples.shape[0]

    def test_each_reference_correct_length(self):
        triples = get_waveform_list(np.array([24]), self.PHASES, self.WIDTHS)
        dref = make_references(self.HEADER, triples)
        for ref in dref.values():
            assert len(ref) == len(self.HEADER)

    def test_keys_are_tuples_of_three(self):
        triples = get_waveform_list(np.array([24]), self.PHASES, self.WIDTHS)
        dref = make_references(self.HEADER, triples)
        for key in dref.keys():
            assert len(key) == 3

    def test_trough_waveform_differs_from_cosine(self):
        triples = get_waveform_list(np.array([24]), self.PHASES, self.WIDTHS)
        dref_cos = make_references(self.HEADER, triples, waveform="cosine")
        dref_tr = make_references(self.HEADER, triples, waveform="trough")
        key = list(dref_cos.keys())[0]
        assert not np.allclose(dref_cos[key], dref_tr[key])


class TestHelpers:
    def test_farctanh_at_zero(self):
        assert farctanh(0.0) == pytest.approx(0.0)

    def test_farctanh_positive(self):
        result = farctanh(0.5)
        assert result == pytest.approx(math.atanh(0.5), rel=1e-4)

    def test_farctanh_clips_near_one(self):
        # Should not blow up at 1.0 (clips to 0.99)
        result = farctanh(1.0)
        assert math.isfinite(result)
        assert result == pytest.approx(math.atanh(0.99), rel=1e-4)

    def test_periodic_in_range(self):
        for x in [-24, -12, 0, 12, 24, 36]:
            result = periodic(float(x))
            assert -12 < result <= 12


class TestGetMatches:
    HEADER = list(range(0, 24, 2))
    PHASES = np.array(list(range(0, 24, 2)))
    WIDTHS = np.array(list(range(2, 24, 2)))

    @pytest.fixture(autouse=True)
    def setup(self):
        triples = get_waveform_list(np.array([24]), self.PHASES, self.WIDTHS)
        self.dref = make_references(self.HEADER, triples)
        self.triple = triples[0]
        self.new_header = tuple(self.HEADER)

    def test_returns_list_of_seven(self):
        kkey = tuple(sorted(range(len(self.HEADER))))
        result = get_matches(kkey, self.triple, self.dref, list(self.new_header))
        assert len(result) == 7

    def test_tau_nonnegative(self):
        # get_matches always returns abs(tau) in position 0
        kkey = tuple(sorted(range(len(self.HEADER))))
        result = get_matches(kkey, self.triple, self.dref, list(self.new_header))
        assert result[0] >= 0.0

    def test_p_value_finite(self):
        kkey = tuple(sorted(range(len(self.HEADER))))
        result = get_matches(kkey, self.triple, self.dref, list(self.new_header))
        assert math.isfinite(result[1])


class TestGetStatProbs:
    """Tests for get_stat_probs(), the main aggregation function.

    This function contains the ``for _ in range(int(...))`` accumulation
    loop that was ``xrange(...)`` in the un-migrated root .pyx source.
    Exercising it here would raise NameError on the unfixed code if tests
    were run against that compiled version.
    """

    HEADER = list(range(0, 24, 2))
    PHASES = np.array(list(range(0, 24, 2)), dtype=float)
    WIDTHS = np.array(list(range(2, 24, 2)), dtype=float)
    PERIODS = np.array([24.0])

    @pytest.fixture(autouse=True)
    def setup(self):
        triples = get_waveform_list(self.PERIODS, self.PHASES, self.WIDTHS)
        self.triples = triples
        self.dref = make_references(self.HEADER, triples)
        self.ref_ranks = rank_references(self.dref, triples)
        self.kkey = tuple(range(len(self.HEADER)))
        self.dorder = {self.kkey: 1.0}

    def test_return_structure(self):
        out1, out2, d_tau, d_per, d_ph, d_na = get_stat_probs(
            self.dorder, self.HEADER, self.triples, self.dref, self.ref_ranks, 10
        )
        assert len(out1) == 6  # [m_per, s_per, m_ph, s_ph, m_na, s_na]
        assert len(out2) == 2  # [m_tau, s_tau]

    def test_mean_tau_finite(self):
        _, out2, *_ = get_stat_probs(
            self.dorder, self.HEADER, self.triples, self.dref, self.ref_ranks, 10
        )
        m_tau, s_tau = out2
        assert math.isfinite(m_tau)
        assert math.isfinite(s_tau)

    def test_distribution_dicts_nonempty(self):
        _, _, d_tau, d_per, d_ph, d_na = get_stat_probs(
            self.dorder, self.HEADER, self.triples, self.dref, self.ref_ranks, 10
        )
        assert len(d_tau) > 0
        assert len(d_per) > 0
        assert len(d_ph) > 0
        assert len(d_na) > 0

    def test_multiple_rank_orders_exercises_accumulation_loop(self):
        """Two entries in dorder forces the range() accumulation loop to run twice."""
        kkey2 = tuple(reversed(range(len(self.HEADER))))
        dorder = {self.kkey: 0.6, kkey2: 0.4}
        out1, out2, *_ = get_stat_probs(
            dorder, self.HEADER, self.triples, self.dref, self.ref_ranks, 20
        )
        assert len(out1) == 6
        assert len(out2) == 2

    def test_distribution_probs_sum_to_one(self):
        out1, out2, d_tau, *_ = get_stat_probs(
            self.dorder, self.HEADER, self.triples, self.dref, self.ref_ranks, 50
        )
        assert sum(d_tau.values()) == pytest.approx(1.0, abs=1e-6)
