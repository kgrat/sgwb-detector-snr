"""The detector networks of arXiv:2502.12828, Sec. II.

CE4020: CE-40 (at CE-A) and CE-20 (at CE-B), one baseline.
ET: the three ET-D interferometers, three baselines.
CE4020ET: all five detectors, ten baselines.
Every pair of detectors in a network is one baseline (T&R Eq. 22).
"""

import itertools

from sgwb_snr.detectors import cosmic_explorer, einstein_telescope
from sgwb_snr.snr import baseline

NETWORKS = ("CE4020", "ET", "CE4020ET")


def network_baselines(network, frequencies_hz):
    """List of (gamma, P_1, P_2) for every detector pair in `network`."""
    if network not in NETWORKS:
        raise ValueError(f"Unknown network '{network}'. Available: {NETWORKS}")
    band_hz = (float(min(frequencies_hz)), float(max(frequencies_hz)))
    # (geometry, noise-curve name) for each detector
    cosmic_explorers = [
        (cosmic_explorer("CE-A", 40.0, *band_hz), "CE-40"),
        (cosmic_explorer("CE-B", 20.0, *band_hz), "CE-20"),
    ]
    einstein = [(ifo, "ET-D") for ifo in einstein_telescope(*band_hz)]
    detectors = {"CE4020": cosmic_explorers, "ET": einstein, "CE4020ET": cosmic_explorers + einstein}[network]
    return [
        baseline(geometry_1, noise_1, geometry_2, noise_2, frequencies_hz)
        for (geometry_1, noise_1), (geometry_2, noise_2) in itertools.combinations(detectors, 2)
    ]
