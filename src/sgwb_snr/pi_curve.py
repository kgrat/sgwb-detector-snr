"""Power-law integrated (PI) sensitivity curves (Thrane & Romano 2013, Sec. III).

For each spectral index beta, the amplitude Omega_beta of the power law
Omega_beta (f / f_ref)^beta that reaches SNR rho over observation time T is
(T&R Eq. 29)

    Omega_beta = rho / sqrt(2T) [int df (f / f_ref)^(2 beta) / Omega_eff^2]^(-1/2),

and the PI curve is the envelope Omega_PI(f) = max_beta Omega_beta (f / f_ref)^beta
(T&R Eq. 30). Any power law tangent to the curve has SNR rho; the curve does not
depend on f_ref. Omega_eff comes from snr.omega_eff.
"""

import numpy as np
from scipy.integrate import trapezoid


def power_law_amplitudes(
    frequencies_hz, omega_eff_values, spectral_indices, reference_frequency_hz, observation_time_s, snr_threshold
):
    """Omega_beta for each beta in spectral_indices (T&R Eq. 29).

    frequencies_hz and omega_eff_values define the integration band and grid.
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    omega_eff_values = np.asarray(omega_eff_values, dtype=float)
    spectral_indices = np.asarray(spectral_indices, dtype=float)
    assert frequencies_hz.ndim == 1 and omega_eff_values.shape == frequencies_hz.shape
    assert spectral_indices.ndim == 1 and spectral_indices.size > 0
    if np.any(frequencies_hz <= 0) or np.any(np.diff(frequencies_hz) <= 0):
        raise ValueError("frequencies_hz must be positive and strictly increasing")
    if np.any(omega_eff_values <= 0):
        raise ValueError("omega_eff must be positive (inf where the network is blind)")
    if observation_time_s <= 0 or snr_threshold <= 0 or reference_frequency_hz <= 0:
        raise ValueError("observation time, SNR threshold and reference frequency must be positive")

    inverse_omega_eff_sq = 1.0 / omega_eff_values**2  # 0 where omega_eff = inf
    amplitudes = np.empty_like(spectral_indices)
    for k, beta in enumerate(spectral_indices):
        integral = trapezoid((frequencies_hz / reference_frequency_hz) ** (2.0 * beta) * inverse_omega_eff_sq, frequencies_hz)
        if not integral > 0:
            raise ValueError(f"network has no sensitivity in the band (beta = {beta})")
        amplitudes[k] = snr_threshold / np.sqrt(2.0 * observation_time_s * integral)
    return amplitudes


def pi_curve(evaluation_frequencies_hz, spectral_indices, amplitudes, reference_frequency_hz):
    """Envelope max_beta Omega_beta (f / f_ref)^beta at evaluation_frequencies_hz (T&R Eq. 30)."""
    evaluation_frequencies_hz = np.asarray(evaluation_frequencies_hz, dtype=float)
    spectral_indices = np.asarray(spectral_indices, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    assert evaluation_frequencies_hz.ndim == 1
    assert spectral_indices.shape == amplitudes.shape
    lines = amplitudes[:, None] * (evaluation_frequencies_hz[None, :] / reference_frequency_hz) ** spectral_indices[:, None]
    return lines.max(axis=0)
