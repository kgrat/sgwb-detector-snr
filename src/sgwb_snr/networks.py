"""The detector networks of arXiv:2502.12828, Sec. II, and the ET design comparison.

CE4020: CE-40 (at CE-A) and CE-20 (at CE-B), one baseline.
ET: the three ET-D interferometers, three baselines.
CE4020ET: all five detectors, ten baselines.
Every pair of detectors in a network is one baseline (T&R Eq. 22).

ET design comparison (Branchesi et al., arXiv:2303.15923), all with the
ET-0304B-22 (CoBA) LF+HF noise curves:
ET-tri10: the triangle with the 10 km curve (three baselines);
ET-2L0, ET-2L45: two 15 km L-shaped detectors, Sardinia and Meuse-Rhine, at
alpha = 0 and 45 deg (one baseline);
CE4020+<design>: the same together with CE-40 and CE-20.
"""

import itertools

from sgwb_snr.detectors import cosmic_explorer, einstein_telescope, einstein_telescope_2l
from sgwb_snr.snr import baseline

NETWORKS = ("CE4020", "ET", "CE4020ET")
ET_DESIGNS = ("ET-tri10", "ET-2L0", "ET-2L45")
DESIGN_NETWORKS = ET_DESIGNS + tuple(f"CE4020+{design}" for design in ET_DESIGNS)
TWO_L_ARM_LENGTH_KM = 15.0  # Branchesi et al. 2L-15km


def _detectors(name, band_hz):
    """(geometry, noise-curve name) for every detector of one network component."""
    if name == "CE4020":
        return [(cosmic_explorer("CE-A", 40.0, *band_hz), "CE-40"), (cosmic_explorer("CE-B", 20.0, *band_hz), "CE-20")]
    if name == "ET":
        return [(ifo, "ET-D") for ifo in einstein_telescope(*band_hz)]
    if name == "ET-tri10":
        return [(ifo, "ET-10km-CoBA") for ifo in einstein_telescope(*band_hz)]
    if name in ("ET-2L0", "ET-2L45"):
        relative_angle_deg = {"ET-2L0": 0.0, "ET-2L45": 45.0}[name]
        return [
            (ifo, "ET-15km-CoBA")
            for ifo in einstein_telescope_2l(relative_angle_deg, TWO_L_ARM_LENGTH_KM, *band_hz)
        ]
    raise ValueError(f"Unknown network component '{name}'")


# Noise curves of the co-located interferometers of an ET triangle: only pairs of
# these can share correlated (e.g. Newtonian) noise.
TRIANGLE_NOISE_CURVES = ("ET-D", "ET-10km-CoBA")


def network_detectors(network, frequencies_hz):
    """List of (geometry, noise-curve name) for every detector in `network`."""
    if network not in NETWORKS + DESIGN_NETWORKS:
        raise ValueError(f"Unknown network '{network}'. Available: {NETWORKS + DESIGN_NETWORKS}")
    band_hz = (float(min(frequencies_hz)), float(max(frequencies_hz)))
    components = {"CE4020ET": ["CE4020", "ET"]}.get(network, network.split("+"))
    return [detector for component in components for detector in _detectors(component, band_hz)]


def colocated_pairs(detectors):
    """Index pairs (i, j), i < j, of interferometers in the same ET triangle."""
    return [
        (i, j)
        for i, j in itertools.combinations(range(len(detectors)), 2)
        if detectors[i][1] in TRIANGLE_NOISE_CURVES and detectors[j][1] in TRIANGLE_NOISE_CURVES
    ]


def network_baselines(network, frequencies_hz):
    """List of (gamma, P_1, P_2) for every detector pair in `network`."""
    detectors = network_detectors(network, frequencies_hz)
    return [
        baseline(geometry_1, noise_1, geometry_2, noise_2, frequencies_hz)
        for (geometry_1, noise_1), (geometry_2, noise_2) in itertools.combinations(detectors, 2)
    ]
