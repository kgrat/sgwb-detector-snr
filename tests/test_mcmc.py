"""Validation of the posterior sampling (arXiv:2502.12828 Eq. 37)."""

import numpy as np
import pytest

from sgwb_snr.fisher import amplitude_covariance, fisher_matrix, string_tension_uncertainty
from sgwb_snr.mcmc import AmplitudePosterior, error_versus_truth, relative_uncertainty, sample, threshold_from_scan
from sgwb_snr.networks import network_baselines
from sgwb_snr.snr import hubble_constant_si
from sgwb_snr.spectra import FOREGROUNDS, REFERENCE_FREQUENCY_HZ, omega_cosmic_strings
from sgwb_snr.spectra import WANG_JIN_ZHANG_2025 as PARAMS

H0_PER_S = hubble_constant_si(PARAMS.h)
YEAR_S = 365.25 * 86400.0
FREQS_HZ = np.arange(5.0, 512.0 + 1e-9, 0.125)
PRIOR_LOG10 = (-18.0, -8.0)  # paper Sec. V
F_STAR = np.array([REFERENCE_FREQUENCY_HZ])


@pytest.fixture(scope="module")
def covariance():
    baselines = network_baselines("CE4020", FREQS_HZ)
    indices = [0.0, FOREGROUNDS["CBC"][1], FOREGROUNDS["CBHE"][1]]
    return amplitude_covariance(fisher_matrix(FREQS_HZ, baselines, indices, YEAR_S, H0_PER_S))


def test_interpolated_template_matches_direct_evaluation(covariance):
    posterior = AmplitudePosterior(np.ones(3), covariance, 1.0 / 3.0, 1.0, PARAMS, PRIOR_LOG10)
    log10_tensions = np.linspace(-17.5, -8.5, 13)
    direct = [omega_cosmic_strings(F_STAR, 10.0**x, 1.0 / 3.0, 1.0, PARAMS)[0] for x in log10_tensions]
    np.testing.assert_allclose(posterior.omega_cs(log10_tensions), direct, rtol=1e-6)


def test_prior_bounds_give_zero_probability(covariance):
    posterior = AmplitudePosterior(np.ones(3), covariance, 1.0 / 3.0, 1.0, PARAMS, PRIOR_LOG10)
    assert np.all(np.isneginf(posterior(np.array([[-19.0, -10.0, -12.0], [-12.0, -7.0, -12.0]]))))


def test_two_component_likelihood_is_the_gaussian_quadratic_form(covariance):
    # Data case 2 model (CS + CBC): log L = -1/2 r^T Sigma^-1 r with r = model - Omega_hat.
    two = covariance[:2, :2]
    omega_hat = np.array([1e-11, 7e-10])
    theta = np.array([-12.0, np.log10(6e-10)])
    posterior = AmplitudePosterior(omega_hat, two, 1.0 / 3.0, 1.0, PARAMS, PRIOR_LOG10)
    residual = np.array([omega_cosmic_strings(F_STAR, 1e-12, 1.0 / 3.0, 1.0, PARAMS)[0], 6e-10]) - omega_hat
    expected = -0.5 * residual @ np.linalg.inv(two) @ residual
    assert posterior(theta)[0] == pytest.approx(expected, rel=1e-5)


def test_error_versus_truth_limits():
    rng = np.random.default_rng(0)
    # Unbiased, median exactly at the truth: reduces to halfwidth / truth.
    samples = np.log10(1e-13) + 0.1 * rng.standard_normal(200001)
    tension = 10.0**samples
    p16, p84 = np.percentile(tension, [16.0, 84.0])
    assert error_versus_truth(samples, 1e-13) == pytest.approx(0.5 * (p84 - p16) / 1e-13, rel=1e-3)
    # Point mass at twice the truth: pure bias, error = 1.
    assert error_versus_truth(np.full(10, np.log10(2e-13)), 1e-13) == pytest.approx(1.0)


def test_gaussian_limit_reproduces_fisher(covariance):
    # With every amplitude measured at >= 20 sigma, the log-uniform priors and the
    # positivity bound are irrelevant and the posterior is Gaussian, so the MCMC
    # Delta(G mu)/(G mu) must equal the Fisher value 2 sigma_CS / Omega_CS.
    sigmas = np.sqrt(np.diag(covariance))
    string_tension = 3.9e-11  # Omega_CS(f*) ~ 40 sigma_CS for CE4020
    truth = np.array([
        omega_cosmic_strings(F_STAR, string_tension, 1.0 / 3.0, 1.0, PARAMS)[0],
        FOREGROUNDS["CBC"][0],
        100.0 * sigmas[2],  # CBHE moved far from the positivity bound
    ])
    assert np.all(truth / sigmas >= 20.0)  # CBC is at 25 sigma for CE4020
    posterior = AmplitudePosterior(truth, covariance, 1.0 / 3.0, 1.0, PARAMS, PRIOR_LOG10)
    samples = sample(posterior, np.log10([string_tension, truth[1], truth[2]]), 32, 100000, seed=1)
    fisher = string_tension_uncertainty(string_tension, sigmas[0], 1.0 / 3.0, 1.0, PARAMS)
    assert relative_uncertainty(samples[:, 0]) == pytest.approx(fisher, rel=0.1)
    assert np.all((samples >= PRIOR_LOG10[0]) & (samples <= PRIOR_LOG10[1]))


def test_threshold_from_scan_interpolates():
    # Delta proportional to G mu^(-1/2): exact power law, so log-log interpolation is exact.
    tensions = np.logspace(-15.0, -12.0, 7)
    deltas = 0.5 * (tensions / 1e-13) ** -0.5
    assert threshold_from_scan(tensions, deltas, 0.5) == pytest.approx(1e-13, rel=1e-10)
    with pytest.raises(ValueError):
        threshold_from_scan(tensions, deltas, 10.0)
    # An unconverged point (inf) next to the crossing must not be interpolated through.
    deltas[3] = np.inf  # the point just below the crossing at index 4
    with pytest.raises(ValueError):
        threshold_from_scan(tensions, deltas, 0.5)
