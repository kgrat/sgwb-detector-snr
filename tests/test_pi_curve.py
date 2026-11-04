"""Validation of power-law integrated curves (Thrane & Romano 2013, Sec. III)."""

import numpy as np
import pytest

from sgwb_snr.networks import ET_DESIGNS, NETWORKS, network_baselines
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
    for name, count in [
        ("CE4020", 1), ("ET", 3), ("CE4020ET", 10),
        ("ET-tri10", 3), ("ET-2L0", 1), ("CE4020+ET-tri10", 10), ("CE4020+ET-2L45", 6),
    ]:
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


def test_et_design_comparison_matches_branchesi():
    # Branchesi et al. (arXiv:2303.15923) Sec. 8.1.1: for stochastic backgrounds the
    # parallel 2L is best below about 100 Hz and the triangle above; the 2L at 45 deg
    # is worse than the parallel 2L below 200-300 Hz and comparable above.
    eff = {design: omega_eff(FREQS_HZ, network_baselines(design, FREQS_HZ), H0_PER_S) for design in ET_DESIGNS}
    crossover_hz = FREQS_HZ[np.argmax(eff["ET-2L0"] > eff["ET-tri10"])]
    assert np.all(eff["ET-2L0"][FREQS_HZ < crossover_hz] < eff["ET-tri10"][FREQS_HZ < crossover_hz])
    assert 70.0 < crossover_hz < 150.0
    at_hz = lambda design, f_hz: eff[design][np.argmin(np.abs(FREQS_HZ - f_hz))]
    assert at_hz("ET-2L45", 20.0) > 5.0 * at_hz("ET-2L0", 20.0)
    assert at_hz("ET-2L45", 300.0) < 2.0 * at_hz("ET-2L0", 300.0)
