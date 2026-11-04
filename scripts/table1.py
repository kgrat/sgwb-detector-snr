"""Reproduce Table I (data case 3) of arXiv:2502.12828 with the Fisher matrix.

For each network and equation of state w: the marginalized error sigma_CS on the
cosmic-string amplitude at f* = 25 Hz, separated from the CBC and CBHE
foregrounds (all three components modelled, data case 3), and the G mu at which
Delta(G mu) / (G mu) = 0.5. 1 yr, 5-512 Hz at 1/8 Hz, T_rd = 1 GeV.
"""

import numpy as np

from sgwb_snr.fisher import amplitude_covariance, detection_threshold, fisher_matrix
from sgwb_snr.networks import NETWORKS, network_baselines
from sgwb_snr.snr import hubble_constant_si
from sgwb_snr.spectra import FOREGROUNDS, cosmic_string_tilt
from sgwb_snr.spectra import WANG_JIN_ZHANG_2025 as PARAMS

H0_PER_S = hubble_constant_si(PARAMS.h)
YEAR_S = 365.25 * 86400.0
FREQS_HZ = np.arange(5.0, 512.0 + 1e-9, 0.125)
T_RD_GEV = 1.0
TARGET = 0.5  # Delta(G mu) / (G mu), paper Sec. V

# Paper Table I, data case 3: G mu at Delta(G mu)/(G mu) = 0.5 for CE4020, ET, CE4020ET.
PAPER_TABLE_I = {
    "1": (1.0, (2.2e-16, 6.3e-17, 2.8e-17)),
    "1/2": (0.5, (8.9e-15, 2.5e-15, 5.6e-16)),
    "1/3": (1.0 / 3.0, (1.4e-13, 1.1e-14, 3.5e-15)),
    "1/4": (0.25, (1.4e-11, 3.5e-13, 1.3e-13)),
}

baselines = {network: network_baselines(network, FREQS_HZ) for network in NETWORKS}
print(f"{'w':>4s}  {'network':9s} {'sigma_CS':>10s} {'G mu (ours)':>12s} {'G mu (paper)':>13s} {'ours/paper':>10s}")
for label, (w, paper_values) in PAPER_TABLE_I.items():
    indices = [cosmic_string_tilt(w), FOREGROUNDS["CBC"][1], FOREGROUNDS["CBHE"][1]]
    for network, paper in zip(NETWORKS, paper_values):
        fisher = fisher_matrix(FREQS_HZ, baselines[network], indices, YEAR_S, H0_PER_S)
        sigma_cs = np.sqrt(amplitude_covariance(fisher)[0, 0])
        threshold = detection_threshold(sigma_cs, w, T_RD_GEV, PARAMS, TARGET, (1e-20, 1e-6))
        print(f"{label:>4s}  {network:9s} {sigma_cs:10.3e} {threshold:12.2e} {paper:13.1e} {threshold / paper:10.2f}")
