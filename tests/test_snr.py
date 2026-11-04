"""Validation of the cross-correlation SNR (Thrane & Romano 2013, arXiv:1310.5300)."""

import itertools

import numpy as np
import pytest

from sgwb_snr.detectors import cosmic_explorer
from sgwb_snr.orf import get_builtin_detector
from sgwb_snr.snr import baseline, hubble_constant_si, omega_eff, snr

H0_PER_S = hubble_constant_si(0.678)  # arXiv:2502.12828 Sec. III (Planck 2018)
YEAR_S = 365.25 * 86400.0
BAND_HZ = (5.0, 512.0)  # analysis band of arXiv:2502.12828


def test_hubble_constant_conversion():
    # 67.8 km/s/Mpc = 67.8 / (3.0857e19 km per Mpc) s^-1.
    assert H0_PER_S == pytest.approx(67.8 / 3.0857e19, rel=1e-4)


def test_constant_noise_flat_spectrum_matches_analytic_integral():
    # gamma, P constant and Omega flat: the integrand is prop. to f^-6, so
    # rho = sqrt(2T) (3 H0^2 / 10 pi^2) |gamma| Omega / P * sqrt((a^-5 - b^-5) / 5).
    a_hz, b_hz = BAND_HZ
    frequencies_hz = np.logspace(np.log10(a_hz), np.log10(b_hz), 20001)
    gamma, psd_per_hz, omega = -0.89, 1e-48, 1e-9
    ones = np.ones_like(frequencies_hz)
    baselines = [(gamma * ones, psd_per_hz * ones, psd_per_hz * ones)]
    expected = (
        np.sqrt(2.0 * YEAR_S)
        * 3.0 * H0_PER_S**2 / (10.0 * np.pi**2)
        * abs(gamma) * omega / psd_per_hz
        * np.sqrt((a_hz**-5 - b_hz**-5) / 5.0)
    )
    rho = snr(frequencies_hz, omega * ones, baselines, YEAR_S, H0_PER_S)
    assert rho == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("n_detectors", [2, 3, 4])
def test_colocated_identical_detectors(n_detectors):
    # T&R Eq. 26: M identical, co-located, co-aligned detectors give
    # S_eff = sqrt(2 / (M (M - 1))) S_n, with S_n = P_n / R and R = Gamma_II = 1/5
    # for an L-shaped detector (T&R Eqs. 16, 17, 27).
    frequencies_hz = np.logspace(1.0, 2.5, 50)
    h1 = get_builtin_detector("H1")
    baselines = [
        baseline(h1, "CE-40", h1, "CE-40", frequencies_hz)
        for _ in itertools.combinations(range(n_detectors), 2)
    ]
    strain_psd_single = baselines[0][1] / (1.0 / 5.0)
    strain_psd_eff = np.sqrt(2.0 / (n_detectors * (n_detectors - 1))) * strain_psd_single
    expected = 2.0 * np.pi**2 * frequencies_hz**3 * strain_psd_eff / (3.0 * H0_PER_S**2)
    np.testing.assert_allclose(omega_eff(frequencies_hz, baselines, H0_PER_S), expected, rtol=1e-6)


def test_snr_equals_omega_eff_form_and_scales():
    # T&R Eqs. 22 and 25: rho = sqrt(2T int (Omega / Omega_eff)^2 df); rho is
    # linear in the amplitude and grows as sqrt(T).
    frequencies_hz = np.linspace(*BAND_HZ, 4000)
    ce40 = cosmic_explorer("CE-A", 40.0, *BAND_HZ)
    ce20 = cosmic_explorer("CE-B", 20.0, *BAND_HZ)
    baselines = [baseline(ce40, "CE-40", ce20, "CE-20", frequencies_hz)]
    omega_gw = 1e-10 * (frequencies_hz / 25.0) ** (2.0 / 3.0)
    rho = snr(frequencies_hz, omega_gw, baselines, YEAR_S, H0_PER_S)
    ratio = omega_gw / omega_eff(frequencies_hz, baselines, H0_PER_S)
    assert rho == pytest.approx(np.sqrt(2.0 * YEAR_S * np.trapezoid(ratio**2, frequencies_hz)), rel=1e-10)
    assert snr(frequencies_hz, 3.0 * omega_gw, baselines, YEAR_S, H0_PER_S) == pytest.approx(3.0 * rho)
    assert snr(frequencies_hz, omega_gw, baselines, 4.0 * YEAR_S, H0_PER_S) == pytest.approx(2.0 * rho)


def test_bad_inputs_rejected():
    frequencies_hz = np.linspace(10.0, 100.0, 10)
    ones = np.ones_like(frequencies_hz)
    good = [(ones, ones, ones)]
    with pytest.raises(ValueError):
        snr(frequencies_hz, ones, good, -1.0, H0_PER_S)
    with pytest.raises(ValueError):
        snr(frequencies_hz, ones, [(ones, -ones, ones)], YEAR_S, H0_PER_S)
    with pytest.raises(AssertionError):
        snr(frequencies_hz, ones[:-1], good, YEAR_S, H0_PER_S)
    with pytest.raises(ValueError):
        hubble_constant_si(67.8)  # km/s/Mpc passed instead of h
