"""Correlated Newtonian noise in the ET triangle and the cosmic-string threshold.

Caporali et al. (arXiv:2501.09057) Eq. 11: every pair of the triangle shares the
noise CSD N_o(f) = r N_d(2.75 Hz) (f / 2.75 Hz)^-8, r in {-0.4, ..., 0.8} (their
injected values). Data case 3 components (CS, CBC, CBHE), Fisher only, 1 yr,
5-512 Hz; G mu at Delta(G mu)/(G mu) = 0.5.

- modelled: r is a free amplitude (tilt fixed), marginalized; the noise
  covariance includes the true N_o (sgwb_snr.correlated).
- unmodelled: the analyst uses the uncorrelated covariance (paper Eq. 25) and
  no r, so Omega_hat is biased by b = fisher.unmodelled_bias; Delta(G mu)/(G mu) =
  sqrt(sigma_CS^2 + b_CS^2) / (k Omega_CS), the Fisher form of the error with
  respect to the truth used for data case 2.

Triangles: ET-D (the paper's ET, CE4020ET) and the 10 km CoBA triangle. The 15 km
parallel 2L (ET-2L0) has no co-located pairs and is printed for comparison.
"""

import numpy as np

from sgwb_snr.correlated import network_fisher
from sgwb_snr.fisher import (
    amplitude_covariance,
    detection_threshold,
    fisher_matrix,
    string_tension_uncertainty,
    unmodelled_bias,
)
from sgwb_snr.networks import colocated_pairs, network_baselines, network_detectors
from sgwb_snr.snr import hubble_constant_si
from sgwb_snr.spectra import FOREGROUNDS, cosmic_string_tilt
from sgwb_snr.spectra import WANG_JIN_ZHANG_2025 as PARAMS

H0_PER_S = hubble_constant_si(PARAMS.h)
YEAR_S = 365.25 * 86400.0
FREQS_HZ = np.arange(5.0, 512.0 + 1e-9, 0.125)
T_RD_GEV = 1.0
TARGET = 0.5
CORRELATIONS = (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8)  # Caporali et al. Sec. V
EQUATIONS_OF_STATE = {"1": 1.0, "1/2": 0.5, "1/3": 1.0 / 3.0, "1/4": 0.25}
TRIANGLE_NETWORKS = ("ET", "CE4020ET", "ET-tri10", "CE4020+ET-tri10")
TWO_L_NETWORKS = ("ET-2L0", "CE4020+ET-2L0")
MAX_TENSION = 1e-10  # report "> 1e-10" beyond, as the paper does for data case 2


def threshold(sigma_cs, w):
    """G mu at Delta = 0.5 as text, or '> 1e-10' if Delta(1e-10) is still above it."""
    if string_tension_uncertainty(MAX_TENSION, sigma_cs, w, T_RD_GEV, PARAMS) > TARGET:
        return f"> {MAX_TENSION:.0e}"
    return f"{detection_threshold(sigma_cs, w, T_RD_GEV, PARAMS, TARGET, (1e-20, MAX_TENSION)):.2e}"


detectors = {network: network_detectors(network, FREQS_HZ) for network in TRIANGLE_NETWORKS}
two_l_baselines = {network: network_baselines(network, FREQS_HZ) for network in TWO_L_NETWORKS}
for label, w in EQUATIONS_OF_STATE.items():
    indices = [cosmic_string_tilt(w), FOREGROUNDS["CBC"][1], FOREGROUNDS["CBHE"][1]]
    two_l = []
    for network in TWO_L_NETWORKS:
        fisher = fisher_matrix(FREQS_HZ, two_l_baselines[network], indices, YEAR_S, H0_PER_S)
        two_l.append(threshold(np.sqrt(amplitude_covariance(fisher)[0, 0]), w))
    print(f"\nw = {label}: G mu threshold (Fisher). No correlated noise: "
          + ", ".join(f"{n} {t}" for n, t in zip(TWO_L_NETWORKS, two_l)))
    print(f"{'r':>5s}  " + " ".join(f"{n + ' ' + kind:>24s}" for n in TRIANGLE_NETWORKS for kind in ("mod", "unmod")), flush=True)
    uncorrelated = {}
    for network in TRIANGLE_NETWORKS:
        pairs = colocated_pairs(detectors[network])
        fisher = network_fisher(FREQS_HZ, detectors[network], pairs, indices, 0.0, YEAR_S, H0_PER_S)
        uncorrelated[network] = (np.sqrt(amplitude_covariance(fisher[:3, :3])[0, 0]), fisher)
    for correlation in CORRELATIONS:
        row = []
        for network in TRIANGLE_NETWORKS:
            pairs = colocated_pairs(detectors[network])
            fisher = network_fisher(FREQS_HZ, detectors[network], pairs, indices, correlation, YEAR_S, H0_PER_S)
            modelled = threshold(np.sqrt(amplitude_covariance(fisher)[0, 0]), w)
            sigma_cs, fisher_0 = uncorrelated[network]
            bias_cs = unmodelled_bias(fisher_0, 3, [correlation])[0]
            unmodelled = threshold(np.hypot(sigma_cs, bias_cs), w)
            row += [modelled, unmodelled]
        print(f"{correlation:5.1f}  " + " ".join(f"{t:>24s}" for t in row), flush=True)
