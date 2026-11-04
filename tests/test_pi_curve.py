"""Validation of power-law integrated curves (Thrane & Romano 2013, Sec. III)."""

import numpy as np
import pytest

from sgwb_snr.networks import NETWORKS, network_baselines
from sgwb_snr.pi_curve import pi_curve, power_law_amplitudes
from sgwb_snr.snr import hubble_constant_si, omega_eff, snr

H0_PER_S = hubble_constant_si(0.678)  # arXiv:2502.12828 Sec. III
YEAR_S = 365.25 * 86400.0
FREQS_HZ = np.arange(5.0, 512.0 + 1e-9, 0.125)  # paper: 5-512 Hz, 1/8 Hz resolution
BETAS = np.arange(-8.0, 9.0)  # T&R Sec. III A: beta = -8, ..., 8
F_REF_HZ = 25.0


@pytest.fixture(scope="module", params=NETWORKS)
def network(request):
    baselines = network_baselines(request.param, FREQS_HZ)
    return baselines, omega_eff(FREQS_HZ, baselines, H0_PER_S)


def test_network_baseline_counts():
    for name, count in [("CE4020", 1), ("ET", 3), ("CE4020ET", 10)]:
        assert len(network_baselines(name, FREQS_HZ[:10])) == count


def test_tangent_power_laws_have_unit_snr(network):
    # T&R Sec. III A: each power law Omega_beta (f / f_ref)^beta has SNR = rho.
    baselines, omega_eff_values = network
    amplitudes = power_law_amplitudes(FREQS_HZ, omega_eff_values, BETAS, F_REF_HZ, YEAR_S, 1.0)
    for beta, amplitude in zip(BETAS, amplitudes):
        omega_gw = amplitude * (FREQS_HZ / F_REF_HZ) ** beta
        assert snr(FREQS_HZ, omega_gw, baselines, YEAR_S, H0_PER_S) == pytest.approx(1.0, rel=1e-9)


def test_flat_spectrum_on_pi_curve_has_unit_snr(network):
    # A flat spectrum touching the PI curve (and nowhere above it) gives SNR = 1.
    baselines, omega_eff_values = network
    amplitudes = power_law_amplitudes(FREQS_HZ, omega_eff_values, BETAS, F_REF_HZ, YEAR_S, 1.0)
    curve = pi_curve(FREQS_HZ, BETAS, amplitudes, F_REF_HZ)
    omega_flat = amplitudes[BETAS == 0.0][0] * np.ones_like(FREQS_HZ)
    assert np.max(omega_flat / curve) == pytest.approx(1.0, rel=1e-12)
    assert snr(FREQS_HZ, omega_flat, baselines, YEAR_S, H0_PER_S) == pytest.approx(1.0, rel=1e-9)


def test_reference_frequency_does_not_matter(network):
    # T&R Eq. 29: f_ref only relabels Omega_beta; the envelope is unchanged.
    _, omega_eff_values = network
    curves = [
        pi_curve(FREQS_HZ, BETAS, power_law_amplitudes(FREQS_HZ, omega_eff_values, BETAS, f_ref, YEAR_S, 1.0), f_ref)
        for f_ref in (25.0, 100.0)
    ]
    np.testing.assert_allclose(curves[0], curves[1], rtol=1e-9)


def test_scales_with_threshold_and_time(network):
    # Omega_beta is proportional to rho / sqrt(T).
    _, omega_eff_values = network
    base = power_law_amplitudes(FREQS_HZ, omega_eff_values, BETAS, F_REF_HZ, YEAR_S, 1.0)
    scaled = power_law_amplitudes(FREQS_HZ, omega_eff_values, BETAS, F_REF_HZ, 4.0 * YEAR_S, 3.0)
    np.testing.assert_allclose(scaled, 1.5 * base, rtol=1e-12)
