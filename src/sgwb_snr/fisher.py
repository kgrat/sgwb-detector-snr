"""Fisher-matrix component separation (arXiv:2502.12828 Sec. IV A; Parida, Mitra
& Jhingan, arXiv:1510.07994).

The background is a sum of components with known shapes,
Omega_GW(f) = sum_a Omega_a F_a(f), F_a = (f / f*)^alpha_a (paper Eq. 19). In the
weak-signal limit the Fisher matrix of the amplitudes Omega_a is (paper
Eqs. 30-31, continuum limit; the same normalization as snr.py)

    F_ab = 2T (3 H0^2 / (10 pi^2))^2 sum_I int df gamma_I^2 F_a F_b / (f^6 P_I1 P_I2),

independent of the amplitudes. Its inverse is the amplitude covariance; the
marginalized error on the cosmic-string amplitude includes the degeneracy with
the foregrounds.

Delta(G mu) / (G mu) follows by linear error propagation through
Omega_CS(f*; G mu), which is what the paper's MCMC (Eq. 37) estimates when the
posterior is close to Gaussian.

If a component present in the data is left out of the model (the paper's data
case 2), the maximum-likelihood estimate (paper Eq. 26, Omega_hat = Gamma^-1 X)
of the modelled amplitudes m is shifted by the unmodelled amplitudes u:

    <Omega_hat_m> = Omega_m + (F_mm)^-1 F_mu Omega_u,

since X = K_m^dagger N^-1 (K_m Omega_m + K_u Omega_u) on average.
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq

from sgwb_snr.snr import inverse_noise_sum
from sgwb_snr.spectra import REFERENCE_FREQUENCY_HZ, omega_cosmic_strings


def fisher_matrix(frequencies_hz, baselines, spectral_indices, observation_time_s, hubble_constant_per_s):
    """F_ab for components (f / f*)^alpha_a, alpha_a in spectral_indices (paper Eqs. 30-31)."""
    if observation_time_s <= 0:
        raise ValueError("observation_time_s must be positive")
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    spectral_indices = np.asarray(spectral_indices, dtype=float)
    assert spectral_indices.ndim == 1 and spectral_indices.size > 0
    inverse_noise = inverse_noise_sum(frequencies_hz, baselines)
    prefactor = 2.0 * observation_time_s * (3.0 * hubble_constant_per_s**2 / (10.0 * np.pi**2)) ** 2
    shapes = (frequencies_hz[None, :] / REFERENCE_FREQUENCY_HZ) ** spectral_indices[:, None]
    weight = inverse_noise / frequencies_hz**6
    fisher = prefactor * trapezoid(shapes[:, None, :] * shapes[None, :, :] * weight, frequencies_hz, axis=-1)
    assert fisher.shape == (spectral_indices.size, spectral_indices.size)
    return fisher


def amplitude_covariance(fisher):
    """Inverse Fisher matrix; raises if the components cannot be separated."""
    fisher = np.asarray(fisher, dtype=float)
    assert fisher.ndim == 2 and fisher.shape[0] == fisher.shape[1]
    # Rescale to unit diagonal before inverting: the amplitudes' natural scales
    # differ by orders of magnitude, which would otherwise mask degeneracy.
    scale = 1.0 / np.sqrt(np.diag(fisher))
    correlation = fisher * scale[:, None] * scale[None, :]
    if np.linalg.cond(correlation) > 1e12:
        raise ValueError("Fisher matrix is singular: component shapes are degenerate")
    covariance = np.linalg.inv(correlation) * scale[:, None] * scale[None, :]
    np.linalg.cholesky(covariance)  # raises unless positive definite
    return covariance


def unmodelled_bias(fisher, n_modelled, unmodelled_amplitudes):
    """Shift (F_mm)^-1 F_mu Omega_u of the first n_modelled amplitudes when the
    remaining components, with true amplitudes unmodelled_amplitudes, are not modelled."""
    fisher = np.asarray(fisher, dtype=float)
    unmodelled_amplitudes = np.asarray(unmodelled_amplitudes, dtype=float)
    assert fisher.ndim == 2 and fisher.shape[0] == fisher.shape[1]
    assert 0 < n_modelled < fisher.shape[0]
    assert unmodelled_amplitudes.shape == (fisher.shape[0] - n_modelled,)
    modelled_covariance = amplitude_covariance(fisher[:n_modelled, :n_modelled])
    bias = modelled_covariance @ fisher[:n_modelled, n_modelled:] @ unmodelled_amplitudes
    assert bias.shape == (n_modelled,)
    return bias


def _log_slope(string_tension, equation_of_state, reheating_temperature_gev, params):
    """k = d ln Omega_CS(f*) / d ln(G mu), by a central difference in ln(G mu)."""
    step = 1e-4
    f_star = np.array([REFERENCE_FREQUENCY_HZ])
    omega_up, omega_down = (
        omega_cosmic_strings(f_star, string_tension * np.exp(sign * step), equation_of_state, reheating_temperature_gev, params)[0]
        for sign in (1.0, -1.0)
    )
    return (np.log(omega_up) - np.log(omega_down)) / (2.0 * step)


def string_tension_uncertainty(string_tension, sigma_cs, equation_of_state, reheating_temperature_gev, params):
    """Delta(G mu) / (G mu) = sigma_CS / (k Omega_CS(f*)), with k the log slope above."""
    if not sigma_cs > 0:
        raise ValueError("sigma_cs must be positive")
    omega = omega_cosmic_strings(
        np.array([REFERENCE_FREQUENCY_HZ]), string_tension, equation_of_state, reheating_temperature_gev, params
    )[0]
    slope = _log_slope(string_tension, equation_of_state, reheating_temperature_gev, params)
    if not slope > 0:
        raise ValueError(f"Omega_CS(f*) does not grow with G mu (slope {slope})")
    return sigma_cs / (slope * omega)


def detection_threshold(sigma_cs, equation_of_state, reheating_temperature_gev, params, target, bracket):
    """G mu at which Delta(G mu) / (G mu) = target (paper: 0.5), searched in bracket = (low, high)."""
    def excess(log10_tension):
        return np.log(
            string_tension_uncertainty(10.0**log10_tension, sigma_cs, equation_of_state, reheating_temperature_gev, params)
            / target
        )
    low, high = np.log10(bracket[0]), np.log10(bracket[1])
    if not excess(low) > 0 > excess(high):
        raise ValueError(f"threshold not bracketed by G mu in {bracket}")
    return 10.0 ** brentq(excess, low, high, xtol=1e-6)
