"""Validation of the detector noise curves (data/noise_curves/SOURCES.md)."""

from pathlib import Path

import numpy as np
import pytest

from sgwb_snr.noise import load_asd, psd

NOISE_DIR = Path(__file__).resolve().parents[1] / "data" / "noise_curves"


def test_et_d_sum_is_xylophone_combination():
    # ET-D = low- plus high-frequency interferometer with independent noise
    # (Hild et al., arXiv:1012.0908): ASD_sum^-2 = ASD_LF^-2 + ASD_HF^-2.
    table = np.loadtxt(NOISE_DIR / "ET-0000A-18_ETDSensitivityCurveTxtFile.txt")
    lf, hf, total = table[:, 1], table[:, 2], table[:, 3]
    np.testing.assert_allclose(total, (lf**-2 + hf**-2) ** -0.5, rtol=1e-6)


@pytest.mark.parametrize("file_name", ["18213_ET10kmcolumns.txt", "18213_ET15kmcolumns.txt"])
def test_coba_column_3_is_xylophone_combination(file_name):
    # ET-0304B-22: columns f, HF, LF, LF+HF, all PSDs. Independent noise in the two
    # instruments gives PSD_sum^-1 = PSD_HF^-1 + PSD_LF^-1 (identifies the columns).
    table = np.loadtxt(NOISE_DIR / file_name)
    hf, lf, total = table[:, 1], table[:, 2], table[:, 3]
    np.testing.assert_allclose(total, 1.0 / (1.0 / hf + 1.0 / lf), rtol=1e-6)
    # HF is the high-frequency instrument: its PSD minimum lies above the LF one.
    assert table[np.argmin(hf), 0] > 100.0 > table[np.argmin(lf), 0]


def test_longer_coba_arms_are_more_sensitive():
    # Arm length 10 -> 15 km lowers the strain noise in the bucket (ET-0304B-22, Fig. 1).
    frequencies_hz = np.array([10.0, 30.0, 100.0, 300.0])
    assert np.all(psd("ET-15km-CoBA", frequencies_hz) < psd("ET-10km-CoBA", frequencies_hz))


@pytest.mark.parametrize("detector", ["CE-40", "CE-20", "ET-D", "ET-10km-CoBA", "ET-15km-CoBA"])
def test_psd_on_tabulated_grid_is_asd_squared(detector):
    frequencies_hz, asd = load_asd(detector)
    np.testing.assert_allclose(psd(detector, frequencies_hz), asd**2, rtol=1e-12)


def test_longer_arms_are_quieter():
    # Strain noise falls with arm length: CE-40 beats CE-20 across the band.
    frequencies_hz = np.logspace(1.0, 3.0, 200)
    assert np.all(psd("CE-40", frequencies_hz) < psd("CE-20", frequencies_hz))


@pytest.mark.parametrize(
    "detector, paper_asd",
    [
        # arXiv:2502.12828 Fig. 1 (left), read off at 100 / 300 / 1000 Hz.
        ("CE-40", [2.147e-25, 2.202e-25, 3.558e-25]),
        ("CE-20", [3.264e-25, 3.176e-25, 4.727e-25]),
        ("ET-D", [3.750e-25, 3.083e-25, 5.594e-25]),
    ],
)
def test_matches_paper_figure_1_above_100_hz(detector, paper_asd):
    # Official files agree with the paper within 10% above 100 Hz; CE sits
    # 4-8% low (newer CE-T2000017 release). See README "Known differences".
    asd = np.sqrt(psd(detector, np.array([100.0, 300.0, 1000.0])))
    np.testing.assert_allclose(asd, paper_asd, rtol=0.10)


def test_no_extrapolation():
    with pytest.raises(ValueError):
        psd("CE-40", np.array([4.99, 10.0]))  # CE files start at 5 Hz


def test_band_edge_accepted():
    # The CE files store 5 Hz as 5.000000000000000888; the paper's band starts
    # at exactly 5 Hz, which must give the tabulated first value.
    _, asd = load_asd("CE-40")
    assert psd("CE-40", np.array([5.0]))[0] == pytest.approx(asd[0] ** 2, rel=1e-12)


def test_unknown_detector_rejected():
    with pytest.raises(ValueError):
        psd("CE", np.array([10.0]))
