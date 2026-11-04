"""Table I (data case 3) of arXiv:2502.12828 from the posterior of Eq. 37.

Same Fisher covariance as scripts/table1.py, but G mu is inferred by MCMC with
the paper's log-uniform priors in [1e-18, 1e-8] on G mu, Omega_CBC and
Omega_CBHE. Omega_hat is set to the injected values (noise-free forecast; the
paper used one simulated noisy realization). For each network and w, G mu is
scanned from 1/100 to 2 times the Fisher threshold, and the threshold is where
Delta(G mu)/(G mu) (half the 16-84% width over the median) first reaches 0.5.
"""

import numpy as np

from sgwb_snr.fisher import amplitude_covariance, detection_threshold, fisher_matrix
from sgwb_snr.mcmc import AmplitudePosterior, NotConverged, relative_uncertainty, sample, threshold_from_scan
from sgwb_snr.networks import NETWORKS, network_baselines
from sgwb_snr.snr import hubble_constant_si
from sgwb_snr.spectra import FOREGROUNDS, REFERENCE_FREQUENCY_HZ, cosmic_string_tilt, omega_cosmic_strings
from sgwb_snr.spectra import WANG_JIN_ZHANG_2025 as PARAMS

H0_PER_S = hubble_constant_si(PARAMS.h)
YEAR_S = 365.25 * 86400.0
FREQS_HZ = np.arange(5.0, 512.0 + 1e-9, 0.125)
T_RD_GEV = 1.0
TARGET = 0.5
PRIOR_LOG10 = (-18.0, -8.0)  # paper Sec. V
# Near the threshold the posterior is a data peak plus a tail to the prior floor and
# tau reaches ~7000 steps (CE4020, w = 1), so 50 tau needs up to ~350000 steps.
N_WALKERS, MAX_STEPS = 32, 1000000
F_STAR = np.array([REFERENCE_FREQUENCY_HZ])

PAPER_TABLE_I = {
    "1": (1.0, (2.2e-16, 6.3e-17, 2.8e-17)),
    "1/2": (0.5, (8.9e-15, 2.5e-15, 5.6e-16)),
    "1/3": (1.0 / 3.0, (1.4e-13, 1.1e-14, 3.5e-15)),
    "1/4": (0.25, (1.4e-11, 3.5e-13, 1.3e-13)),
}

baselines = {network: network_baselines(network, FREQS_HZ) for network in NETWORKS}
print(f"{'w':>4s}  {'network':9s} {'Fisher':>9s} {'MCMC':>9s} {'paper':>8s} {'MCMC/paper':>10s}", flush=True)
for label, (w, paper_values) in PAPER_TABLE_I.items():
    indices = [cosmic_string_tilt(w), FOREGROUNDS["CBC"][1], FOREGROUNDS["CBHE"][1]]
    for network, paper in zip(NETWORKS, paper_values):
        covariance = amplitude_covariance(fisher_matrix(FREQS_HZ, baselines[network], indices, YEAR_S, H0_PER_S))
        fisher_threshold = detection_threshold(np.sqrt(covariance[0, 0]), w, T_RD_GEV, PARAMS, TARGET, (1e-20, 1e-6))
        tensions = np.logspace(np.log10(fisher_threshold / 100.0), np.log10(2.0 * fisher_threshold), 12)
        tensions = tensions[(tensions > 10.0 ** PRIOR_LOG10[0]) & (tensions < 10.0 ** PRIOR_LOG10[1])]
        deltas, unconverged = [], 0
        for seed, string_tension in enumerate(tensions):
            truth = np.array([
                omega_cosmic_strings(F_STAR, string_tension, w, T_RD_GEV, PARAMS)[0],
                FOREGROUNDS["CBC"][0],
                FOREGROUNDS["CBHE"][0],
            ])
            posterior = AmplitudePosterior(truth, covariance, w, T_RD_GEV, PARAMS, PRIOR_LOG10)
            try:
                samples = sample(posterior, np.log10([string_tension, truth[1], truth[2]]), N_WALKERS, MAX_STEPS, seed)
                deltas.append(relative_uncertainty(samples[:, 0]))
            except NotConverged:
                # Counted as "not measured"; reported below so it is never silent.
                deltas.append(np.inf)
                unconverged += 1
        threshold = threshold_from_scan(tensions, deltas, TARGET)
        note = f"  ({unconverged} unconverged scan points)" if unconverged else ""
        print(f"{label:>4s}  {network:9s} {fisher_threshold:9.2e} {threshold:9.2e} {paper:8.1e} {threshold / paper:10.2f}{note}", flush=True)
