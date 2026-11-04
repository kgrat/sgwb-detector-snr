"""Component separation with correlated noise between co-located ET interferometers.

The interferometers of the ET triangle share test-mass sites a few hundred metres
apart, so seismic and Newtonian noise (NN) correlate between them (Janssens et
al., arXiv:2206.06809). Following Caporali et al. (arXiv:2501.09057, Eq. 11),
the noise cross-spectral density (CSD) of every pair in the triangle is

    N_o(f) = r N_d(f_p) (f / f_p)^n,   f_p = 2.75 Hz, n = -8 (body-wave NN),

with N_d the one-sided PSD, real r (phase 0 or pi), and -1/2 <= N_o/N_d <= 1
for a positive-definite noise matrix. Pairs at different sites are uncorrelated.

Each baseline p = (I, J) measures the cross-spectrum
<C_p(f)> = (3 H0^2 / 10 pi^2) gamma_p(f) Omega_GW(f) / f^3 + N_o,p(f)
(paper Eq. 18 plus the noise CSD). In the weak-signal limit the cross-spectra
of all baselines are Gaussian with covariance (Isserlis' theorem)

    Sigma_pq(f) proportional to N_IK N_JL + N_IL N_JK,   p = (I, J), q = (K, L),

which reduces to the paper's Eq. 25, P_I P_J on the diagonal and 0 elsewhere,
when N is diagonal. Two baselines sharing a detector are then correlated at
first order in N_o. The Fisher matrix of amplitudes with per-baseline templates
h_a,p(f) (expected cross-spectrum per unit amplitude) is

    F_ab = 2T int df h_a(f)^T Sigma(f)^-1 h_b(f),

the same normalization as fisher.fisher_matrix, which it equals for N_o = 0.
The CSD amplitude r can be added as one more component, with template N_o/r on
the correlated pairs; leaving it out of the model biases the others
(fisher.unmodelled_bias).
"""

import itertools

import numpy as np
from scipy.integrate import trapezoid

from sgwb_snr.noise import psd
from sgwb_snr.orf import tensor_orf
from sgwb_snr.spectra import REFERENCE_FREQUENCY_HZ

NEWTONIAN_NOISE_PIVOT_HZ = 2.75  # Caporali et al. Eq. 11
NEWTONIAN_NOISE_INDEX = -8.0  # body-wave NN (Caporali et al. Sec. V; Janssens et al.)


def newtonian_noise_template(frequencies_hz, pivot_psd_per_hz):
    """N_o(f) / r = N_d(f_p) (f / f_p)^n of Caporali et al. Eq. 11, in 1/Hz."""
    if not pivot_psd_per_hz > 0:
        raise ValueError("pivot_psd_per_hz must be positive")
    return pivot_psd_per_hz * (np.asarray(frequencies_hz, dtype=float) / NEWTONIAN_NOISE_PIVOT_HZ) ** NEWTONIAN_NOISE_INDEX


def detector_noise_matrix(psds_per_hz, correlated_pairs, csd_per_hz):
    """N_IJ(f), shape (n_f, n_det, n_det): PSDs on the diagonal, csd_per_hz on the
    correlated pairs, zero elsewhere. Raises unless positive definite at every f."""
    psds_per_hz = np.asarray(psds_per_hz, dtype=float)
    n_det, n_f = psds_per_hz.shape
    csd_per_hz = np.asarray(csd_per_hz, dtype=float)
    assert csd_per_hz.shape == (n_f,)
    noise = np.zeros((n_f, n_det, n_det))
    noise[:, np.arange(n_det), np.arange(n_det)] = psds_per_hz.T
    for i, j in correlated_pairs:
        assert 0 <= i < j < n_det
        noise[:, i, j] = noise[:, j, i] = csd_per_hz
    if np.any(np.linalg.eigvalsh(noise)[:, 0] <= 0):
        raise ValueError("noise matrix is not positive definite (Caporali et al.: need -1/2 <= N_o/N_d <= 1)")
    return noise


def cross_spectrum_covariance(noise_matrix, pairs):
    """Sigma_pq = N_IK N_JL + N_IL N_JK for baselines p = (I, J), q = (K, L); shape (n_f, n_b, n_b)."""
    first = np.array([i for i, _ in pairs])
    second = np.array([j for _, j in pairs])
    n_ik = noise_matrix[:, first[:, None], first[None, :]]
    n_jl = noise_matrix[:, second[:, None], second[None, :]]
    n_il = noise_matrix[:, first[:, None], second[None, :]]
    n_jk = noise_matrix[:, second[:, None], first[None, :]]
    return n_ik * n_jl + n_il * n_jk


def gravitational_wave_templates(frequencies_hz, orfs, spectral_indices, hubble_constant_per_s):
    """h_a,p(f) = (3 H0^2 / 10 pi^2) gamma_p(f) (f/f*)^alpha_a / f^3; shape (n_comp, n_b, n_f)."""
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    orfs = np.asarray(orfs, dtype=float)
    spectral_indices = np.asarray(spectral_indices, dtype=float)
    assert orfs.ndim == 2 and orfs.shape[1] == frequencies_hz.size and spectral_indices.ndim == 1
    shapes = (frequencies_hz / REFERENCE_FREQUENCY_HZ) ** spectral_indices[:, None] / frequencies_hz**3
    return 3.0 * hubble_constant_per_s**2 / (10.0 * np.pi**2) * orfs[None, :, :] * shapes[:, None, :]


def fisher_from_templates(frequencies_hz, templates, covariance, observation_time_s):
    """F_ab = 2T int df h_a^T Sigma^-1 h_b for templates (n_comp, n_b, n_f), Sigma (n_f, n_b, n_b)."""
    if observation_time_s <= 0:
        raise ValueError("observation_time_s must be positive")
    templates = np.asarray(templates, dtype=float)
    n_comp, n_b, n_f = templates.shape
    assert covariance.shape == (n_f, n_b, n_b)
    weighted = np.linalg.solve(covariance, np.moveaxis(templates, 0, -1).transpose(1, 0, 2))  # (n_f, n_b, n_comp)
    integrand = np.einsum("apf,fpb->abf", templates, weighted)
    fisher = 2.0 * observation_time_s * trapezoid(integrand, frequencies_hz, axis=-1)
    assert fisher.shape == (n_comp, n_comp)
    return 0.5 * (fisher + fisher.T)


def network_fisher(frequencies_hz, detectors, correlated_pairs, spectral_indices, correlation,
                   observation_time_s, hubble_constant_per_s):
    """Fisher matrix of (power-law amplitudes..., r) for a network with Caporali correlated noise.

    detectors: list of (geometry, noise-curve name); correlated_pairs: index pairs
    sharing the CSD; correlation: the true r, which sets the noise covariance. The
    last component is r itself, with template N_o/r on the correlated pairs.
    The triangle interferometers must share one noise curve (Caporali et al. Eq. 10).
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    triangle_curves = {detectors[k][1] for pair in correlated_pairs for k in pair}
    if len(triangle_curves) != 1:
        raise ValueError(f"correlated pairs must share one noise curve, got {sorted(triangle_curves)}")
    unit_csd_per_hz = newtonian_noise_template(
        frequencies_hz, psd(triangle_curves.pop(), np.array([NEWTONIAN_NOISE_PIVOT_HZ]))[0]
    )
    psds_per_hz = np.array([psd(noise, frequencies_hz) for _, noise in detectors])
    noise = detector_noise_matrix(psds_per_hz, correlated_pairs, correlation * unit_csd_per_hz)
    pairs = list(itertools.combinations(range(len(detectors)), 2))
    orfs = np.array([tensor_orf(detectors[i][0], detectors[j][0], frequencies_hz) for i, j in pairs])
    gw = gravitational_wave_templates(frequencies_hz, orfs, spectral_indices, hubble_constant_per_s)
    noise_template = np.array([unit_csd_per_hz if pair in correlated_pairs else np.zeros_like(frequencies_hz) for pair in pairs])
    templates = np.concatenate([gw, noise_template[None]], axis=0)
    return fisher_from_templates(frequencies_hz, templates, cross_spectrum_covariance(noise, pairs), observation_time_s)
