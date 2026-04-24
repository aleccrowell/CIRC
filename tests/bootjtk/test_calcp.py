"""Tests for CalcP.py functions."""
import numpy as np
import pytest

import circ.bootjtk.CalcP as CalcP


class TestEmpP:
    def test_value_above_null_gets_low_p(self):
        # Data tau=0.9 is larger than all null taus → small empirical p
        taus = np.array([0.9])
        emps = np.array([0.1, 0.2, 0.3, 0.1, 0.0])
        ps = CalcP.empP(taus, emps)
        assert ps[0] < 0.5

    def test_value_below_null_gets_high_p(self):
        # Data tau=0.0 is at or below most null taus → large empirical p
        taus = np.array([0.0])
        emps = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        ps = CalcP.empP(taus, emps)
        assert ps[0] > 0.5

    def test_output_length_matches_input(self):
        taus = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        emps = np.linspace(0, 1, 100)
        ps = CalcP.empP(taus, emps)
        assert len(ps) == len(taus)

    def test_probabilities_in_valid_range(self):
        taus = np.linspace(0, 1, 20)
        emps = np.linspace(0, 1, 50)
        ps = CalcP.empP(taus, emps)
        assert all(0.0 < p <= 1.0 for p in ps)

    def test_known_value(self):
        # tau=1.0 exceeds all of [0.1, 0.2, 0.3, 0.4] → p = (0+1)/(4+1) = 0.2
        taus = np.array([1.0])
        emps = np.array([0.1, 0.2, 0.3, 0.4])
        ps = CalcP.empP(taus, emps)
        assert ps[0] == pytest.approx(1.0 / 5.0)

    def test_monotone_decreasing(self):
        # Higher tau should yield lower (or equal) empirical p
        emps = np.array([0.2, 0.4, 0.6, 0.3, 0.5])
        taus = np.array([0.1, 0.5, 0.9])
        ps = CalcP.empP(taus, emps)
        assert ps[0] >= ps[1] >= ps[2]


class TestPrepare:
    def test_returns_five_items(self):
        taus = np.random.default_rng(0).normal(0.4, 0.1, 200)
        result = CalcP.prepare(taus)
        assert len(result) == 5

    def test_keys_are_sorted(self):
        taus = np.random.default_rng(1).normal(0.5, 0.15, 300)
        keys, *_ = CalcP.prepare(taus)
        assert list(keys) == sorted(keys)

    def test_intvalues_sum_to_one(self):
        taus = np.random.default_rng(2).normal(0.5, 0.1, 200)
        _, intvalues, *_ = CalcP.prepare(taus)
        assert sum(intvalues) == pytest.approx(1.0, abs=1e-6)

    def test_p0_has_three_gamma_params(self):
        taus = np.random.default_rng(3).normal(0.5, 0.1, 200)
        _, _, _, p0, _ = CalcP.prepare(taus)
        assert len(p0) == 3

    def test_limit_is_float(self):
        taus = np.random.default_rng(4).normal(0.5, 0.1, 200)
        _, _, _, _, limit = CalcP.prepare(taus)
        assert isinstance(limit, float)

    def test_yerr_same_length_as_keys(self):
        taus = np.random.default_rng(5).normal(0.5, 0.1, 200)
        keys, _, yerr, _, _ = CalcP.prepare(taus)
        assert len(yerr) == len(keys)
