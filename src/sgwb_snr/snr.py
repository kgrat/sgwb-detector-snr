"""Cross-correlation SNR for an isotropic, unpolarized SGWB (Thrane & Romano 2013).

Thrane & Romano (arXiv:1310.5300) write the network SNR (their Eq. 22) with the
unnormalized ORF Gamma_IJ and the GW strain PSD S_h = 3 H0^2 Omega / (2 pi^2 f^3)
(their Eq. 3). Our normalized ORF gamma (orf.py) obeys Gamma = gamma / 5 for any
opening angles (both reduce to D_1 : D_2 up to fixed factors), so

    rho^2 = 2 T sum_{I<J} int df gamma_IJ^2 (3 H0^2 / (10 pi^2))^2
                                 Omega^2 / (f^6 P_I P_J),

the same normalization as the Fisher matrix of arXiv:2502.12828, Eqs. 30-31.
Assumes weak signal (Omega small compared with the noise) and uncorrelated noise
in every detector pair, including the co-located ET interferometers.
"""

import astropy.units as u
import numpy as np
from scipy.integrate import trapezoid

from sgwb_snr.noise import psd
from sgwb_snr.orf import tensor_orf


def hubble_constant_si(h):
    """H0 = 100 h km/s/Mpc, in 1/s."""
    if not 0.0 < h < 2.0:
        raise ValueError(f"unphysical dimensionless Hubble parameter h = {h}")
    return (100.0 * h * u.km / u.s / u.Mpc).to_value(1 / u.s)


def baseline(detector_1, noise_1, detector_2, noise_2, frequencies_hz):
    """(gamma, P_1, P_2) on frequencies_hz for one detector pair.

    detector_k: bilby Interferometer (geometry); noise_k: noise.py curve name.
    """
    return (
        tensor_orf(detector_1, detector_2, frequencies_hz),
        psd(noise_1, frequencies_hz),
        psd(noise_2, frequencies_hz),
    )


def _inverse_noise_sum(frequencies_hz, baselines):
    """sum_{I<J} gamma_IJ^2 / (P_I P_J), the network's combined inverse noise."""
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    assert frequencies_hz.ndim == 1, "frequencies_hz must be 1D"
    if np.any(frequencies_hz <= 0) or np.any(np.diff(frequencies_hz) <= 0):
        raise ValueError("frequencies_hz must be positive and strictly increasing")
    if len(baselines) == 0:
        raise ValueError("need at least one baseline")
    total = np.zeros_like(frequencies_hz)
    for gamma, psd_1, psd_2 in baselines:
        for array in (gamma, psd_1, psd_2):
            assert np.shape(array) == frequencies_hz.shape, "baseline arrays must match frequencies_hz"
        if np.any(psd_1 <= 0) or np.any(psd_2 <= 0):
            raise ValueError("noise PSDs must be positive")
        total += np.asarray(gamma) ** 2 / (np.asarray(psd_1) * np.asarray(psd_2))
    return total


def omega_eff(frequencies_hz, baselines, hubble_constant_per_s):
    """Effective network sensitivity in Omega units (T&R Eqs. 3, 23).

    S_eff = [sum Gamma^2 / (P_I P_J)]^(-1/2) = 5 [sum gamma^2 / (P_I P_J)]^(-1/2),
    Omega_eff = 2 pi^2 f^3 S_eff / (3 H0^2). Infinite where every gamma vanishes.
    """
    inverse = _inverse_noise_sum(frequencies_hz, baselines)
    strain_psd_eff = np.full_like(inverse, np.inf)
    np.divide(5.0, np.sqrt(inverse), out=strain_psd_eff, where=inverse > 0)
    return 2.0 * np.pi**2 * np.asarray(frequencies_hz) ** 3 * strain_psd_eff / (3.0 * hubble_constant_per_s**2)


def snr(frequencies_hz, omega_gw, baselines, observation_time_s, hubble_constant_per_s):
    """Expected network SNR of a background Omega_gw(f) (T&R Eq. 22).

    The frequency integral is a trapezoid rule over frequencies_hz, so the
    grid sets both the band [f_min, f_max] and the integration accuracy.
    """
    if observation_time_s <= 0:
        raise ValueError("observation_time_s must be positive")
    omega_gw = np.asarray(omega_gw, dtype=float)
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    assert omega_gw.shape == frequencies_hz.shape, "omega_gw must match frequencies_hz"
    inverse = _inverse_noise_sum(frequencies_hz, baselines)
    prefactor = (3.0 * hubble_constant_per_s**2 / (10.0 * np.pi**2)) ** 2
    integrand = prefactor * omega_gw**2 * inverse / frequencies_hz**6
    return np.sqrt(2.0 * observation_time_s * trapezoid(integrand, frequencies_hz))
