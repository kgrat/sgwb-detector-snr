"""Table I (data case 2) of arXiv:2502.12828: CBHE in the data but not in the model.

The data contain CS + CBC + CBHE; the separation models only CS + CBC (paper
Sec. IV B). Omega_hat is the noise-free maximum-likelihood estimate of the two
modelled amplitudes, i.e. the injected values shifted by the Fisher bias
(F_mm)^-1 F_mu Omega_CBHE (fisher.unmodelled_bias); its covariance is the
inverse of the CS + CBC block of the Fisher matrix. G mu is then inferred by MCMC
with the paper's log-uniform priors in [1e-18, 1e-8].

Delta(G mu)/(G mu) is the total error with respect to the injected G mu,
sqrt((median - truth)^2 + halfwidth^2) / truth (mcmc.error_versus_truth),
because the posterior width alone is blind to the bias; the
case-3 definition (halfwidth / median) is printed alongside. G mu is scanned in
steps of 0.25 dex from 1/10 of the data case 3 Fisher threshold up to 1e-10,
where the paper's scan evidently stopped (it reports "> 1e-10").

Usage: python scripts/table1_case2.py [w labels, e.g. 1 1/3]  (default: all)
"""

import sys

import numpy as np

from sgwb_snr.fisher import amplitude_covariance, detection_threshold, fisher_matrix, unmodelled_bias
from sgwb_snr.mcmc import (
    AmplitudePosterior,
    NotConverged,
    error_versus_truth,
    relative_uncertainty,
    sample,
    threshold_from_scan,
)
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
SCAN_TOP_LOG10, SCAN_STEP_DEX = -10.0, 0.25
N_WALKERS, MAX_STEPS = 32, 1000000  # cap as in table1_mcmc.py
F_STAR = np.array([REFERENCE_FREQUENCY_HZ])

# Paper Table I, data case 2 (CE4020, ET, CE4020ET); None means "> 1e-10".
PAPER_TABLE_I_CASE_2 = {
    "1": (1.0, (6.3e-16, 2.0e-15, 5.6e-15)),
    "1/2": (0.5, (5.6e-14, 3.2e-14, 2.0e-14)),
    "1/3": (1.0 / 3.0, (5.6e-13, 2.0e-13, 5.0e-14)),
    "1/4": (0.25, (None, None, None)),
}


def format_threshold(tensions, deltas, below_at_start_allowed):
    """Threshold from the scan, or '> top' if Delta never reaches the target.

    With below_at_start_allowed, Delta <= target already at the first point gives
    '< start'. This is expected only for the bias-blind width/median measure when
    the bias is positive (w = 1): the posterior then confidently measures the
    leaked CBHE power. For the headline measure it would mean the scan started
    too high, so threshold_from_scan raises.
    """
    if below_at_start_allowed and deltas[0] <= TARGET:
        return f"< {tensions[0]:.1e}", None
    if np.min(deltas) > TARGET:
        return f"> {tensions[-1]:.1e}", None
    threshold = threshold_from_scan(tensions, deltas, TARGET)
    return f"{threshold:.2e}", threshold


labels = sys.argv[1:] or list(PAPER_TABLE_I_CASE_2)
unknown = set(labels) - set(PAPER_TABLE_I_CASE_2)
if unknown:
    raise ValueError(f"unknown w labels {sorted(unknown)}; choose from {list(PAPER_TABLE_I_CASE_2)}")

baselines = {network: network_baselines(network, FREQS_HZ) for network in NETWORKS}
print(f"{'w':>4s}  {'network':9s} {'bias/sigma':>10s} {'vs truth':>10s} {'width/med':>10s} {'paper':>9s}", flush=True)
for label in labels:
    w, paper_values = PAPER_TABLE_I_CASE_2[label]
    indices = [cosmic_string_tilt(w), FOREGROUNDS["CBC"][1], FOREGROUNDS["CBHE"][1]]
    for network, paper in zip(NETWORKS, paper_values):
        fisher = fisher_matrix(FREQS_HZ, baselines[network], indices, YEAR_S, H0_PER_S)
        covariance = amplitude_covariance(fisher[:2, :2])
        bias = unmodelled_bias(fisher, 2, [FOREGROUNDS["CBHE"][0]])
        case_3_threshold = detection_threshold(
            np.sqrt(amplitude_covariance(fisher)[0, 0]), w, T_RD_GEV, PARAMS, TARGET, (1e-20, 1e-6)
        )
        start_log10 = np.log10(case_3_threshold / 10.0)
        n_points = int(np.ceil((SCAN_TOP_LOG10 - start_log10) / SCAN_STEP_DEX)) + 1
        all_tensions = np.logspace(start_log10, SCAN_TOP_LOG10, n_points)
        versus_truth, width_over_median = [], []
        for seed, string_tension in enumerate(all_tensions):
            truth = np.array([omega_cosmic_strings(F_STAR, string_tension, w, T_RD_GEV, PARAMS)[0], FOREGROUNDS["CBC"][0]])
            posterior = AmplitudePosterior(truth + bias, covariance, w, T_RD_GEV, PARAMS, PRIOR_LOG10)
            try:
                samples = sample(posterior, np.log10([string_tension, truth[1]]), N_WALKERS, MAX_STEPS, seed)
                versus_truth.append(error_versus_truth(samples[:, 0], string_tension))
                width_over_median.append(relative_uncertainty(samples[:, 0]))
            except NotConverged:
                # Counted as "not measured"; threshold_from_scan raises if it brackets the target.
                versus_truth.append(np.inf)
                width_over_median.append(np.inf)
            if min(versus_truth) <= TARGET and min(width_over_median) <= TARGET:
                break  # both definitions have crossed; larger G mu cannot change the first crossing
        tensions = all_tensions[: len(versus_truth)]
        unconverged = int(np.sum(np.isinf(versus_truth)))
        truth_text, _ = format_threshold(tensions, np.array(versus_truth), below_at_start_allowed=False)
        width_text, _ = format_threshold(tensions, np.array(width_over_median), below_at_start_allowed=True)
        paper_text = "> 1.0e-10" if paper is None else f"{paper:.1e}"
        note = f"  ({unconverged} unconverged scan points)" if unconverged else ""
        print(
            f"{label:>4s}  {network:9s} {bias[0] / np.sqrt(covariance[0, 0]):+10.1f} {truth_text:>10s} "
            f"{width_text:>10s} {paper_text:>9s}{note}",
            flush=True,
        )
