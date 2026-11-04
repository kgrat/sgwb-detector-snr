"""Validation of the correlated-noise Fisher matrix (Caporali et al., arXiv:2501.09057)."""

import itertools

import numpy as np
import pytest

from sgwb_snr.correlated import (
    cross_spectrum_covariance,
    detector_noise_matrix,
    fisher_from_templates,
    gravitational_wave_templates,
    network_fisher,
)
from sgwb_snr.fisher import fisher_matrix, unmodelled_bias
from sgwb_snr.noise import psd
from sgwb_snr.networks import colocated_pairs, network_baselines, network_detectors
from sgwb_snr.orf import tensor_orf
from sgwb_snr.snr import hubble_constant_si
from sgwb_snr.spectra import FOREGROUNDS

H0_PER_S = hubble_constant_si(0.678)
YEAR_S = 365.25 * 86400.0
FREQS_HZ = np.arange(5.0, 512.0 + 1e-9, 0.125)
DATA_CASE_3 = [0.0, FOREGROUNDS["CBC"][1], FOREGROUNDS["CBHE"][1]]


@pytest.fixture(scope="module")
def ce4020et():
    detectors = network_detectors("CE4020ET", FREQS_HZ)
    return detectors, colocated_pairs(detectors)


def test_colocated_pairs_are_the_et_triangle(ce4020et):
    detectors, pairs = ce4020et
    assert [d[1] for d in detectors] == ["CE-40", "CE-20", "ET-D", "ET-D", "ET-D"]
    assert pairs == [(2, 3), (2, 4), (3, 4)]


def test_uncorrelated_limit_is_the_paper_fisher_matrix(ce4020et):
    # r = 0: Sigma is diagonal with P_I P_J (paper Eq. 25), so the GW block must equal
    # fisher.fisher_matrix, the formula behind every earlier result.
    detectors, pairs = ce4020et
    fisher = network_fisher(FREQS_HZ, detectors, pairs, DATA_CASE_3, 0.0, YEAR_S, H0_PER_S)
    expected = fisher_matrix(FREQS_HZ, network_baselines("CE4020ET", FREQS_HZ), DATA_CASE_3, YEAR_S, H0_PER_S)
    np.testing.assert_allclose(fisher[:3, :3], expected, rtol=1e-8)


def test_isserlis_covariance_of_baselines_sharing_a_detector():
    # Three detectors, PSD 2 and CSD 0.5 on every pair: Sigma_(01),(01) = N_00 N_11 + N_01^2
    # and Sigma_(01),(02) = N_00 N_12 + N_02 N_10 (first order in the CSD).
    noise = detector_noise_matrix(np.full((3, 1), 2.0), [(0, 1), (0, 2), (1, 2)], np.array([0.5]))
    sigma = cross_spectrum_covariance(noise, [(0, 1), (0, 2), (1, 2)])[0]
    assert sigma[0, 0] == pytest.approx(2.0 * 2.0 + 0.5**2)
    assert sigma[0, 1] == pytest.approx(2.0 * 0.5 + 0.5 * 0.5)


@pytest.mark.parametrize("correlation", [2.0, -0.9])
def test_unphysical_correlation_raises(ce4020et, correlation):
    # Per unit r, N_o/N_d = 0.63 at 5 Hz for ET-D: r = 2 gives 1.26 > 1 and
    # r = -0.9 gives -0.57 < -1/2 (Caporali et al. Sec. III A bound).
    detectors, pairs = ce4020et
    with pytest.raises(ValueError):
        network_fisher(FREQS_HZ, detectors, pairs, DATA_CASE_3, correlation, YEAR_S, H0_PER_S)


def test_unmodelled_copy_is_absorbed_with_correlated_covariance(ce4020et):
    # An unmodelled per-baseline template identical to the CBC one must be absorbed
    # exactly into Omega_CBC, also with a correlated (non-diagonal) covariance.
    detectors, pairs = ce4020et
    all_pairs = list(itertools.combinations(range(len(detectors)), 2))
    orfs = np.array([tensor_orf(detectors[i][0], detectors[j][0], FREQS_HZ) for i, j in all_pairs])
    gw = gravitational_wave_templates(FREQS_HZ, orfs, DATA_CASE_3, H0_PER_S)
    psds = np.array([psd(noise, FREQS_HZ) for _, noise in detectors])
    noise = detector_noise_matrix(psds, pairs, 0.3 * psds[2] * (FREQS_HZ / 5.0) ** -8)
    fisher = fisher_from_templates(FREQS_HZ, np.concatenate([gw, gw[1:2]]), cross_spectrum_covariance(noise, all_pairs), YEAR_S)
    bias = unmodelled_bias(fisher, 3, [2e-10])
    np.testing.assert_allclose(bias, [0.0, 2e-10, 0.0], atol=1e-6 * 2e-10)


def test_positive_correlation_mimics_negative_omega():
    # ET pairs have gamma ~ -3/8: a positive noise CSD adds to C_p like a negative
    # Omega. Unmodelled, it must pull a single (CBC-shaped) amplitude down.
    detectors = network_detectors("ET", FREQS_HZ)
    pairs = colocated_pairs(detectors)
    for correlation in (0.4, -0.4):
        fisher = network_fisher(FREQS_HZ, detectors, pairs, [FOREGROUNDS["CBC"][1]], correlation, YEAR_S, H0_PER_S)
        assert np.sign(unmodelled_bias(fisher, 1, [correlation])[0]) == -np.sign(correlation)
