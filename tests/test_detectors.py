"""Physical validation of the CE/ET geometry (Gupta et al., arXiv:2307.10421)."""

import itertools

import numpy as np
import pytest

from sgwb_snr.detectors import (
    _ET_2L_SITES,
    cosmic_explorer,
    einstein_telescope,
    einstein_telescope_2l,
    great_circle_azimuth_deg,
)
from sgwb_snr.orf import tensor_orf

BAND_HZ = (5.0, 512.0)  # analysis band of arXiv:2502.12828
LOW_FREQ_HZ = np.array([0.1])


@pytest.fixture(scope="module")
def et():
    return einstein_telescope(*BAND_HZ)


@pytest.fixture(scope="module")
def ce40():
    return cosmic_explorer("CE-A", 40.0, *BAND_HZ)


@pytest.fixture(scope="module")
def ce20():
    return cosmic_explorer("CE-B", 20.0, *BAND_HZ)


def _unit(vector):
    return vector / np.linalg.norm(vector)


def test_ce40_ce20_matches_paper_figure_1(ce40, ce20):
    # arXiv:2502.12828 Fig. 1 (middle panel), CE-40 - CE-20 curve, read off
    # the figure at 1 Hz: gamma = +0.134 (reading precision ~0.003).
    gamma = tensor_orf(ce40, ce20, np.array([1.0]))
    assert gamma[0] == pytest.approx(0.134, abs=0.003)


def test_ce_is_right_handed_l(ce40, ce20):
    # Table II caption: x, y and the local outward vertical are right-handed.
    for ce in (ce40, ce20):
        assert np.degrees(np.arccos(np.dot(ce.x, ce.y))) == pytest.approx(90.0, abs=1e-9)
        # Geodetic and geocentric verticals differ by < 0.2 deg.
        assert np.dot(np.cross(ce.x, ce.y), _unit(ce.vertex)) > 0.9999


def test_unknown_ce_configuration_rejected():
    # Gupta et al.: CE-B hosts only a 20 km detector.
    with pytest.raises(ValueError):
        cosmic_explorer("CE-B", 40.0, *BAND_HZ)


def test_et_is_closed_equilateral_triangle(et):
    # Walking three 10 km x-arms must return to ET1's vertex.
    for i, j in itertools.combinations(range(3), 2):
        side_m = np.linalg.norm(et[i].vertex - et[j].vertex)
        assert side_m == pytest.approx(10e3, abs=50.0)


def test_et_arms_lie_along_triangle_edges(et):
    # Each interferometer's two arms point at the other two vertices.
    for i, ifo in enumerate(et):
        assert np.degrees(np.arccos(np.dot(ifo.x, ifo.y))) == pytest.approx(60.0, abs=1e-9)
        edges = [_unit(et[j].vertex - ifo.vertex) for j in range(3) if j != i]
        for arm in (ifo.x, ifo.y):
            assert max(abs(np.dot(arm, edge)) for edge in edges) > 0.9999


def test_et_pair_low_frequency_limit(et):
    # Two coplanar 60-degree interferometers rotated by 120 degrees:
    # D_1 : D_2 = (3/8) cos(240 deg), so gamma(f -> 0) = 2 D_1 : D_2 = -3/8.
    # Also compare with 2 D_1 : D_2 from bilby's tensors, independent of pygwb.
    for a, b in itertools.combinations(et, 2):
        gamma = tensor_orf(a, b, LOW_FREQ_HZ)[0]
        assert gamma == pytest.approx(-0.375, abs=0.002)
        assert gamma == pytest.approx(2.0 * np.sum(a.detector_tensor * b.detector_tensor), abs=1e-3)


def test_et_null_stream(et, ce40, ce20):
    # The three ET detector tensors sum to zero (120-degree rotations of one
    # tensor), so at low frequency sum_k gamma(ETk, X) = 0 for any X.
    for other in (ce40, ce20):
        total = sum(tensor_orf(ifo, other, LOW_FREQ_HZ)[0] for ifo in et)
        assert total == pytest.approx(0.0, abs=0.005)


def _two_l_orf(relative_angle_deg, frequencies_hz):
    return tensor_orf(*einstein_telescope_2l(relative_angle_deg, 15.0, *BAND_HZ), frequencies_hz)


@pytest.fixture(scope="module")
def parallel_angle_deg():
    # Arms parallel with respect to the great circle joining the sites: the angle
    # of each x-arm from the great-circle direction (Sardinia -> Meuse-Rhine,
    # continued past Meuse-Rhine) must be equal.
    s, m = _ET_2L_SITES["Sardinia"], _ET_2L_SITES["Meuse-Rhine"]
    outbound_deg = great_circle_azimuth_deg(s["latitude_deg"], s["longitude_deg"], m["latitude_deg"], m["longitude_deg"])
    inbound_deg = great_circle_azimuth_deg(m["latitude_deg"], m["longitude_deg"], s["latitude_deg"], s["longitude_deg"])
    return (inbound_deg + 180.0 - outbound_deg) % 180.0


def test_two_l_parallel_angle_matches_branchesi(parallel_angle_deg):
    # Branchesi et al. Sec. 2: alpha = beta + 2.51 deg.
    assert parallel_angle_deg == pytest.approx(2.51, abs=0.005)


def test_two_l_orf_peaks_at_great_circle_parallel(parallel_angle_deg):
    # In the long-wavelength limit gamma is largest for great-circle-parallel arms.
    # Rotating the Meuse-Rhine L the wrong way would put the peak at -2.51 deg.
    alphas_deg = np.arange(-10.0, 10.0 + 1e-9, 0.01)
    gammas = [_two_l_orf(alpha, LOW_FREQ_HZ)[0] for alpha in alphas_deg]
    assert alphas_deg[np.argmax(gammas)] == pytest.approx(parallel_angle_deg, abs=0.015)


def test_two_l_orf_vanishes_45_degrees_from_parallel(parallel_angle_deg):
    # Branchesi et al. Sec. 2: at beta = 45 deg gamma -> 0 as f * dx -> 0
    # (f dx / c = 4e-4 at 0.1 Hz); alpha = 45 deg keeps a small residual.
    assert abs(_two_l_orf(parallel_angle_deg + 45.0, LOW_FREQ_HZ)[0]) < 1e-3
    assert 0.05 < _two_l_orf(45.0, LOW_FREQ_HZ)[0] < 0.15
