"""Validation of the Fisher component separation (arXiv:2502.12828 Sec. IV A)."""

import numpy as np
import pytest

from sgwb_snr.fisher import amplitude_covariance, detection_threshold, fisher_matrix, string_tension_uncertainty
from sgwb_snr.networks import network_baselines
from sgwb_snr.pi_curve import power_law_amplitudes
from sgwb_snr.snr import hubble_constant_si, omega_eff
from sgwb_snr.spectra import FOREGROUNDS, REFERENCE_FREQUENCY_HZ, cosmic_string_tilt, omega_cosmic_strings
from sgwb_snr.spectra import WANG_JIN_ZHANG_2025 as PARAMS

H0_PER_S = hubble_constant_si(PARAMS.h)
YEAR_S = 365.25 * 86400.0
FREQS_HZ = np.arange(5.0, 512.0 + 1e-9, 0.125)  # paper: 5-512 Hz, 1/8 Hz
DATA_CASE_3 = [0.0, FOREGROUNDS["CBC"][1], FOREGROUNDS["CBHE"][1]]  # CS (w = 1/3), CBC, CBHE
F_STAR = np.array([REFERENCE_FREQUENCY_HZ])


@pytest.fixture(scope="module")
def ce4020():
    return network_baselines("CE4020", FREQS_HZ)


@pytest.mark.parametrize("index", DATA_CASE_3)
def test_single_component_error_is_unit_snr_amplitude(ce4020, index):
    # One component: sigma = 1 / sqrt(F_00), and an amplitude sigma has SNR 1,
    # i.e. the T&R Eq. 29 amplitude Omega_beta for rho = 1 (pi_curve.py).
    sigma = np.sqrt(1.0 / fisher_matrix(FREQS_HZ, ce4020, [index], YEAR_S, H0_PER_S)[0, 0])
    unit_snr = power_law_amplitudes(
        FREQS_HZ, omega_eff(FREQS_HZ, ce4020, H0_PER_S), [index], REFERENCE_FREQUENCY_HZ, YEAR_S, 1.0
    )[0]
    assert sigma == pytest.approx(unit_snr, rel=1e-10)


def test_marginalizing_over_foregrounds_increases_error(ce4020):
    # Cramer-Rao: the marginalized error can only grow when components are added.
    alone = np.sqrt(amplitude_covariance(fisher_matrix(FREQS_HZ, ce4020, [0.0], YEAR_S, H0_PER_S))[0, 0])
    with_cbc = np.sqrt(amplitude_covariance(fisher_matrix(FREQS_HZ, ce4020, DATA_CASE_3[:2], YEAR_S, H0_PER_S))[0, 0])
    full = np.sqrt(amplitude_covariance(fisher_matrix(FREQS_HZ, ce4020, DATA_CASE_3, YEAR_S, H0_PER_S))[0, 0])
    assert alone < with_cbc < full


def test_covariance_inverts_fisher(ce4020):
    fisher = fisher_matrix(FREQS_HZ, ce4020, DATA_CASE_3, YEAR_S, H0_PER_S)
    np.testing.assert_allclose(fisher, fisher.T, rtol=1e-12)
    identity = amplitude_covariance(fisher) @ fisher
    np.testing.assert_allclose(identity, np.eye(3), atol=1e-8)


def test_degenerate_components_rejected(ce4020):
    with pytest.raises(ValueError):
        amplitude_covariance(fisher_matrix(FREQS_HZ, ce4020, [0.0, 0.0], YEAR_S, H0_PER_S))


def test_plateau_error_propagation():
    # w = 1/3 plateau: Omega_CS proportional to sqrt(G mu), so
    # Delta(G mu)/(G mu) = 2 Delta(Omega_CS)/Omega_CS (radiation-era plateau).
    string_tension, sigma = 1e-13, 1e-12
    omega = omega_cosmic_strings(F_STAR, string_tension, 1.0 / 3.0, 1.0, PARAMS)[0]
    ratio = string_tension_uncertainty(string_tension, sigma, 1.0 / 3.0, 1.0, PARAMS)
    assert ratio == pytest.approx(2.0 * sigma / omega, rel=3e-3)


@pytest.mark.parametrize("w", [1.0, 0.5, 0.25])
def test_non_standard_log_slope(w):
    # Above f_rd, Omega_CS(f*) = Omega_CS(f_rd) (f*/f_rd)^alpha_CS with
    # Omega_CS(f_rd) ~ sqrt(G mu) and f_rd ~ (G mu)^(-1/2) (paper Eqs. 7, 9), so
    # Omega_CS(f*) ~ (G mu)^((1 + alpha_CS)/2) while f_rd sits on the plateau.
    string_tension, sigma = 1e-10, 1e-12
    omega = omega_cosmic_strings(F_STAR, string_tension, w, 1.0, PARAMS)[0]
    slope = (1.0 + cosmic_string_tilt(w)) / 2.0
    ratio = string_tension_uncertainty(string_tension, sigma, w, 1.0, PARAMS)
    assert ratio == pytest.approx(sigma / (slope * omega), rel=1e-3)


def test_detection_threshold_solves_target():
    sigma = 1e-12
    threshold = detection_threshold(sigma, 1.0 / 3.0, 1.0, PARAMS, 0.5, (1e-18, 1e-8))
    assert string_tension_uncertainty(threshold, sigma, 1.0 / 3.0, 1.0, PARAMS) == pytest.approx(0.5, rel=1e-5)
    # On the plateau this is Omega_CS(f*) = 4 sigma.
    assert omega_cosmic_strings(F_STAR, threshold, 1.0 / 3.0, 1.0, PARAMS)[0] == pytest.approx(4.0 * sigma, rel=3e-3)
