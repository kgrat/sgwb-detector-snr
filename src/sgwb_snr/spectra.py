"""Omega_GW(f) of the three components in arXiv:2502.12828, Sec. III.

Cosmic strings (CS), standard cosmology: paper Eqs. 3-6, i.e. the analytic
template of Sousa, Avelino & Guedes (arXiv:2002.01079, Eqs. 45, 53, 58):
loops formed and decaying in the radiation era, formed in the radiation era and
decaying in the matter era, and formed in the matter era. Valid for loops born
larger than Gamma G mu t.

Loop size: the paper's alpha = 0.1 is read as loops born
with length l = 0.1 t, so eps_r = 0.1 / (Gamma G mu) (Sousa et al.'s alpha xi_r
= 0.1). The printed Eq. 3 (eps_r = alpha xi_r / (Gamma G mu) with alpha = 0.1)
gives a plateau 1.93x lower than the paper's Figs. 2-3; this reading reproduces
them to 0.5% and matches the F = 0.1, l = 0.1 t calibration behind A_r = 0.54
(Sousa et al., Sec. V). eps_m = eps_r xi_m / xi_r as in the paper.

Non-standard cosmology (equation of state w before T_rd): paper Eqs. 7-9.
Above f_rd the spectrum is Omega_CS(f_rd) (f / f_rd)^(-d); Fig. 2 confirms the
pivot at f_rd.

Foregrounds: power laws Omega (f / f*)^index (paper Eqs. 12 and 14).

Frequencies in Hz, H0 in 1/s (natural units, c = 1).
"""

from dataclasses import dataclass

import numpy as np

from sgwb_snr.snr import hubble_constant_si

REFERENCE_FREQUENCY_HZ = 25.0  # f* of arXiv:2502.12828

# (amplitude at f*, spectral index): paper Eq. 12 (CBC) and Eq. 14 (CBHE).
FOREGROUNDS = {
    "CBC": (7.50e-10, 2.0 / 3.0),
    "CBHE": (6.84e-12, 9.0 / 5.0),
}


@dataclass(frozen=True)
class CosmicStringParameters:
    loop_size_over_time: float  # l = this * t at loop birth
    gw_emission_efficiency: float  # Gamma
    xi_radiation: float  # L = xi t in the radiation era (VOS scaling)
    xi_matter: float
    a_radiation: float  # loop-distribution normalization A_r
    a_matter: float
    h: float  # H0 = 100 h km/s/Mpc
    omega_matter: float  # today
    omega_radiation: float


# arXiv:2502.12828 Sec. III A (Planck 2018 cosmology), with l = 0.1 t.
WANG_JIN_ZHANG_2025 = CosmicStringParameters(
    loop_size_over_time=0.1,
    gw_emission_efficiency=50.0,
    xi_radiation=0.271,
    xi_matter=0.625,
    a_radiation=0.54,
    a_matter=0.039,
    h=0.678,
    omega_matter=0.308,
    omega_radiation=9.1476e-5,
)


def power_law(frequencies_hz, amplitude, spectral_index):
    """amplitude * (f / f*)^spectral_index with f* = 25 Hz."""
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    return amplitude * (frequencies_hz / REFERENCE_FREQUENCY_HZ) ** spectral_index


def foreground(name, frequencies_hz):
    """CBC or CBHE foreground spectrum (paper Eqs. 12, 14)."""
    if name not in FOREGROUNDS:
        raise ValueError(f"Unknown foreground '{name}'. Available: {sorted(FOREGROUNDS)}")
    return power_law(frequencies_hz, *FOREGROUNDS[name])


def _check_inputs(frequencies_hz, string_tension, params):
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    assert frequencies_hz.ndim == 1, "frequencies_hz must be 1D"
    if np.any(frequencies_hz <= 0):
        raise ValueError("frequencies_hz must be positive")
    if not string_tension > 0:
        raise ValueError(f"G mu must be positive, got {string_tension}")
    if params.loop_size_over_time < params.gw_emission_efficiency * string_tension:
        raise ValueError("template valid only for loops born larger than Gamma G mu t")
    return frequencies_hz


def omega_cosmic_strings_standard(frequencies_hz, string_tension, params):
    """Omega_CS(f) in standard cosmology: paper Eqs. 3-6 (Sousa et al. Eqs. 45, 53, 58)."""
    f = _check_inputs(frequencies_hz, string_tension, params)
    h0 = hubble_constant_si(params.h)
    gamma_gmu = params.gw_emission_efficiency * string_tension
    om, orad = params.omega_matter, params.omega_radiation
    eps_r = params.loop_size_over_time / gamma_gmu
    eps_m = eps_r * params.xi_matter / params.xi_radiation
    b_r = 4.0 * h0 * np.sqrt(orad) / gamma_gmu
    b_m = 3.0 * h0 * np.sqrt(om) / gamma_gmu

    # Eq. 3: loops formed and decaying in the radiation era.
    radiation = (
        128.0 / 9.0 * np.pi * params.a_radiation * orad * string_tension / eps_r
        * ((f * (1.0 + eps_r) / (b_r * om / orad + f)) ** 1.5 - 1.0)
    )

    # Eq. 4: formed in the radiation era, decaying in the matter era.
    ratio = om / orad
    shifted = b_m * np.sqrt(ratio) + f
    radiation_matter = (
        32.0 * np.sqrt(3.0) * np.pi * (om * orad) ** 0.75 * h0 * params.a_radiation / params.gw_emission_efficiency
        * (eps_r + 1.0) ** 1.5 / (np.sqrt(f) * eps_r)
        * (
            ratio**0.25 / np.sqrt(shifted) * (2.0 + f / shifted)
            - 1.0 / np.sqrt(b_m + f) * (2.0 + f / (b_m + f))
        )
    )

    # Eq. 5: loops formed in the matter era.
    matter = (
        54.0 * np.pi * h0 * om**1.5 * params.a_matter / params.gw_emission_efficiency
        * (eps_m + 1.0) / eps_m * b_m / f
        * (
            (2.0 * b_m + f) / (b_m * (b_m + f))
            - (2.0 * eps_m + 1.0) / (f * eps_m * (eps_m + 1.0))
            + 2.0 / f * (np.log1p(1.0 / eps_m) + np.log(b_m / (b_m + f)))
        )
    )

    total = radiation + radiation_matter + matter
    assert total.shape == f.shape and np.all(np.isfinite(total))
    if np.any(total <= 0):
        raise ValueError("non-positive Omega_CS: outside the template's validity")
    return total


def reheating_frequency_hz(string_tension, reheating_temperature_gev, params):
    """f_rd (paper Eq. 9), above which a non-standard era modifies the spectrum."""
    if not reheating_temperature_gev > 0 or not string_tension > 0:
        raise ValueError("T_rd and G mu must be positive")
    return (
        8.67e-3 * reheating_temperature_gev
        * np.sqrt(1e-12 / (params.loop_size_over_time * string_tension))
    )


def cosmic_string_tilt(equation_of_state):
    """alpha_CS = -d of paper Eq. 8, the spectral index above f_rd, for w > 1/9."""
    w = equation_of_state
    if not 1.0 / 9.0 < w <= 1.0:
        # Paper Eq. 8 gives d = 1 for w <= 1/9, but its text says the spectrum is
        # then unchanged; we do not resolve this and only allow 1/9 < w <= 1.
        raise ValueError(f"equation of state w = {w} outside (1/9, 1]")
    return 2.0 * (3.0 * w - 1.0) / (3.0 * w + 1.0)


def omega_cosmic_strings(frequencies_hz, string_tension, equation_of_state, reheating_temperature_gev, params):
    """Omega_CS*(f) for an era with equation of state w before T_rd (paper Eq. 7).

    w = 1/3 is the standard cosmology (paper Eqs. 3-6).
    """
    f = _check_inputs(frequencies_hz, string_tension, params)
    tilt = cosmic_string_tilt(equation_of_state)
    standard = omega_cosmic_strings_standard(f, string_tension, params)
    if tilt == 0.0:
        return standard
    f_rd = reheating_frequency_hz(string_tension, reheating_temperature_gev, params)
    at_f_rd = omega_cosmic_strings_standard(np.array([f_rd]), string_tension, params)[0]
    return np.where(f <= f_rd, standard, at_f_rd * (f / f_rd) ** tilt)
