"""Validation of the source spectra (arXiv:2502.12828 Sec. III; Sousa et al. 2020)."""

import numpy as np
import pytest

from sgwb_snr.spectra import (
    FOREGROUNDS,
    REFERENCE_FREQUENCY_HZ,
    WANG_JIN_ZHANG_2025 as PARAMS,
    cosmic_string_tilt,
    foreground,
    omega_cosmic_strings,
    omega_cosmic_strings_standard,
    reheating_frequency_hz,
)

F_STAR = np.array([REFERENCE_FREQUENCY_HZ])
BAND_HZ = np.linspace(5.0, 512.0, 500)  # analysis band of the paper


@pytest.mark.parametrize("string_tension", [1e-16, 1e-13, 1e-10])
def test_plateau_scales_as_sqrt_string_tension(string_tension):
    # Radiation-era plateau, alpha >> Gamma G mu: Omega_CS proportional to
    # sqrt(G mu) (Sousa et al. Eq. 45 for eps_r >> 1), so Delta(G mu)/(G mu)
    # ~ 2 Delta(Omega_CS)/Omega_CS. The matter-era 1/f tail adds 0.24% at
    # 25 Hz for G mu = 1e-16 (ten times less per decade of G mu), hence 3e-3.
    low = omega_cosmic_strings_standard(F_STAR, string_tension, PARAMS)[0]
    high = omega_cosmic_strings_standard(F_STAR, 4.0 * string_tension, PARAMS)[0]
    assert high / low == pytest.approx(2.0, rel=3e-3)


def test_plateau_matches_analytic_limit():
    # f >> B_r Omega_m / Omega_r: Eq. 3 tends to
    # (128/9) pi A_r Omega_r G mu [(1 + eps_r)^(3/2) - 1] / eps_r.
    string_tension = 1e-13
    eps_r = PARAMS.loop_size_over_time / (PARAMS.gw_emission_efficiency * string_tension)
    plateau = (
        128.0 / 9.0 * np.pi * PARAMS.a_radiation * PARAMS.omega_radiation
        * string_tension * ((1.0 + eps_r) ** 1.5 - 1.0) / eps_r
    )
    assert omega_cosmic_strings_standard(F_STAR, string_tension, PARAMS)[0] == pytest.approx(plateau, rel=1e-4)


@pytest.mark.parametrize(
    "string_tension, frequency_hz, paper_value",
    [
        # arXiv:2502.12828 Fig. 2, w = 1/3 curves (figure read-off).
        (1e-10, 300.0, 9.92e-10),
        (1e-16, 1000.0, 1.03e-12),
        # Gmu = 1e-16 below f_rd: matter-era bump and its tail.
        (1e-16, 0.1, 1.56e-12),
        (1e-16, 1.0, 1.06e-12),
    ],
)
def test_standard_spectrum_matches_paper_figure_2(string_tension, frequency_hz, paper_value):
    omega = omega_cosmic_strings_standard(np.array([frequency_hz]), string_tension, PARAMS)[0]
    assert omega == pytest.approx(paper_value, rel=0.06)


@pytest.mark.parametrize(
    "string_tension, w, frequency_hz, paper_value",
    [
        # arXiv:2502.12828 Fig. 2, non-standard curves (figure read-off).
        (1e-10, 1.0, 0.1, 3.554e-8),
        (1e-10, 0.5, 0.1, 4.133e-9),
        (1e-10, 0.25, 0.1, 3.543e-10),
        (1e-16, 1.0, 1000.0, 3.750e-10),
        (1e-16, 0.5, 1000.0, 1.092e-11),
        (1e-16, 0.25, 1000.0, 1.908e-13),
    ],
)
def test_non_standard_spectrum_matches_paper_figure_2(string_tension, w, frequency_hz, paper_value):
    omega = omega_cosmic_strings(np.array([frequency_hz]), string_tension, w, 1.0, PARAMS)[0]
    assert omega == pytest.approx(paper_value, rel=0.06)


def test_reheating_frequency_below_band():
    # Paper Sec. III A 2: for T_rd = 1 GeV and G mu >= 1e-16, f_rd < 5 Hz.
    assert reheating_frequency_hz(1e-16, 1.0, PARAMS) < 5.0


@pytest.mark.parametrize("w, tilt", [(1.0, 1.0), (0.5, 0.4), (1.0 / 3.0, 0.0), (0.25, -2.0 / 7.0)])
def test_tilt_and_in_band_power_law(w, tilt):
    # Paper Eqs. 8 and 10: above f_rd, Omega_CS = Omega_CS(f*) (f / f*)^alpha_CS.
    assert cosmic_string_tilt(w) == pytest.approx(tilt, abs=1e-12)
    if tilt != 0.0:
        omega = omega_cosmic_strings(BAND_HZ, 1e-13, w, 1.0, PARAMS)
        slope = np.diff(np.log(omega)) / np.diff(np.log(BAND_HZ))
        np.testing.assert_allclose(slope, tilt, rtol=1e-9)


def test_foregrounds():
    # Paper Eqs. 12 and 14: amplitude at f* = 25 Hz and spectral index.
    for name, (amplitude, index) in FOREGROUNDS.items():
        assert foreground(name, F_STAR)[0] == pytest.approx(amplitude, rel=1e-12)
        assert foreground(name, np.array([50.0]))[0] / amplitude == pytest.approx(2.0**index, rel=1e-12)


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        omega_cosmic_strings_standard(F_STAR, 0.0, PARAMS)
    with pytest.raises(ValueError):
        omega_cosmic_strings_standard(F_STAR, 1e-2, PARAMS)  # loops smaller than Gamma G mu t
    with pytest.raises(ValueError):
        cosmic_string_tilt(0.1)  # w <= 1/9: paper Eq. 8 and its text disagree
    with pytest.raises(ValueError):
        foreground("BNS", F_STAR)
