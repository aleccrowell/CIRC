"""Tests for model functions and fitting routines in bootjtk/arbfit.py."""
import numpy as np
import pytest

import circ.bootjtk.arbfit as arbfit


class TestLinear:
    def test_positive_slope(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        result = arbfit.line(x, [2.0, 1.0])
        expected = np.array([1.0, 3.0, 5.0, 7.0])
        np.testing.assert_allclose(result, expected)

    def test_zero_slope(self):
        x = np.array([0.0, 5.0, 10.0])
        result = arbfit.line(x, [0.0, 4.0])
        np.testing.assert_allclose(result, [4.0, 4.0, 4.0])

    def test_negative_slope(self):
        x = np.array([1.0, 2.0, 3.0])
        result = arbfit.line(x, [-1.0, 5.0])
        np.testing.assert_allclose(result, [4.0, 3.0, 2.0])


class TestGaussian:
    def test_peak_at_mean(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        p = [1.0, 2.0, 0.5, 0.0]  # scale=1, mean=2, std=0.5, offset=0
        result = arbfit.gauss(x, p)
        assert result[2] == pytest.approx(max(result))  # peak at index 2 (x=2)

    def test_offset(self):
        x = np.array([0.0])
        p = [0.0, 0.0, 1.0, 3.5]  # scale=0 → just offset
        result = arbfit.gauss(x, p)
        assert result[0] == pytest.approx(3.5)

    def test_symmetry(self):
        x = np.array([-1.0, 0.0, 1.0])
        p = [1.0, 0.0, 1.0, 0.0]
        result = arbfit.gauss(x, p)
        assert result[0] == pytest.approx(result[2])


class TestExponential:
    def test_at_zero(self):
        x = np.array([0.0])
        p = [2.0, 1.0, 0.0]  # 2*exp(0) + 0 = 2
        assert arbfit.exp(x, p)[0] == pytest.approx(2.0)

    def test_decays_to_offset(self):
        x = np.array([1000.0])
        p = [1.0, 5.0, 0.5]  # 1*exp(-5000) + 0.5 ≈ 0.5
        result = arbfit.exp(x, p)[0]
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_monotone_decreasing(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        p = [1.0, 1.0, 0.0]
        result = arbfit.exp(x, p)
        assert all(result[i] > result[i+1] for i in range(len(result)-1))


class TestPowerLaw:
    def test_linear(self):
        # p[1]=1 → linear
        x = np.array([2.0, 3.0, 4.0])
        result = arbfit.plaw(x, [1.0, 1.0])
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])

    def test_square(self):
        x = np.array([2.0, 3.0])
        result = arbfit.plaw(x, [1.0, 2.0])
        np.testing.assert_allclose(result, [4.0, 9.0])

    def test_scaling(self):
        x = np.array([1.0])
        result = arbfit.plaw(x, [5.0, 3.0])
        assert result[0] == pytest.approx(5.0)


class TestSine:
    def test_periodicity(self):
        x = np.array([0.0, 1.0])
        p = [1.0, 1.0, 0.0, 0.0]  # period=1
        r0 = arbfit.sine(x, p)
        x_next = np.array([1.0, 2.0])
        r1 = arbfit.sine(x_next, p)
        np.testing.assert_allclose(r0, r1, atol=1e-10)

    def test_offset(self):
        x = np.array([0.0])
        p = [0.0, 1.0, 0.0, 7.0]  # scale=0 → just offset
        assert arbfit.sine(x, p)[0] == pytest.approx(7.0)


class TestArbFit:
    def test_fits_linear_data(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.1, 30)
        p0 = np.array([1.0, 0.0])
        x2, par, xfit, yfit = arbfit.arbFit(fct=arbfit.line, x=x, y=y, p0=p0)
        slope, intercept = par[0]
        assert slope == pytest.approx(2.0, abs=0.2)
        assert intercept == pytest.approx(1.0, abs=0.5)

    def test_fit_returns_four_items(self):
        x = np.linspace(0, 5, 20)
        y = 3.0 * x + 0.5
        result = arbfit.arbFit(fct=arbfit.line, x=x, y=y, p0=np.array([1.0, 0.0]))
        assert len(result) == 4

    def test_xfit_yfit_same_length(self):
        x = np.linspace(0, 10, 15)
        y = 1.5 * x + 2.0
        _, _, xfit, yfit = arbfit.arbFit(fct=arbfit.line, x=x, y=y, p0=np.array([1.0, 0.0]))
        assert len(xfit) == len(yfit)

    def test_xfit_spans_data_range(self):
        x = np.linspace(1.0, 9.0, 20)
        y = x * 0.5
        _, _, xfit, _ = arbfit.arbFit(fct=arbfit.line, x=x, y=y, p0=np.array([0.5, 0.0]))
        assert xfit[0] == pytest.approx(x.min(), rel=1e-3)
        assert xfit[-1] == pytest.approx(x.max(), rel=1e-3)

    def test_fits_gaussian_data(self):
        np.random.seed(7)
        x = np.linspace(-3, 3, 50)
        y = 2.0 * np.exp(-0.5 * x**2) + np.random.normal(0, 0.05, 50)
        p0 = np.array([1.5, 0.0, 1.0, 0.0])
        _, par, _, _ = arbfit.arbFit(fct=arbfit.gauss, x=x, y=y, p0=p0)
        scale = par[0][0]
        mean = par[0][1]
        assert scale == pytest.approx(2.0, abs=0.3)
        assert mean == pytest.approx(0.0, abs=0.2)
