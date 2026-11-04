"""ET triangle vs 2L (Branchesi et al., arXiv:2303.15923) for the cosmic-string search.

1. 1-sigma PI curves (1 yr, 5-512 Hz) of the ET-D triangle and of the three
   design-comparison options with the ET-0304B-22 (CoBA) noise curves: the 10 km
   triangle and the 15 km 2L at alpha = 0 and 45 deg. Compare with Branchesi
   et al. Fig. 27 (which integrates over a wider band).
2. Table I (data case 3, Fisher): G mu at Delta(G mu)/(G mu) = 0.5 for each ET
   option alone and together with CE4020. Same settings as scripts/table1.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sgwb_snr.fisher import amplitude_covariance, detection_threshold, fisher_matrix
from sgwb_snr.networks import DESIGN_NETWORKS, network_baselines
from sgwb_snr.pi_curve import pi_curve, power_law_amplitudes
from sgwb_snr.snr import hubble_constant_si, omega_eff
from sgwb_snr.spectra import FOREGROUNDS, REFERENCE_FREQUENCY_HZ, cosmic_string_tilt
from sgwb_snr.spectra import WANG_JIN_ZHANG_2025 as PARAMS

H0_PER_S = hubble_constant_si(PARAMS.h)
YEAR_S = 365.25 * 86400.0
FREQS_HZ = np.arange(5.0, 512.0 + 1e-9, 0.125)
BETAS = np.round(np.arange(-8.0, 8.0 + 1e-9, 0.01), 2)
T_RD_GEV = 1.0
TARGET = 0.5
EQUATIONS_OF_STATE = {"1": 1.0, "1/2": 0.5, "1/3": 1.0 / 3.0, "1/4": 0.25}
LABELS = {
    "ET": "ET-D triangle (10 km)",
    "ET-tri10": "triangle 10 km (CoBA)",
    "ET-2L0": r"2L 15 km, $\alpha$ = 0$^\circ$ (CoBA)",
    "ET-2L45": r"2L 15 km, $\alpha$ = 45$^\circ$ (CoBA)",
}

baselines = {network: network_baselines(network, FREQS_HZ) for network in ("ET", "CE4020ET") + DESIGN_NETWORKS}

fig, ax = plt.subplots(figsize=(6, 4.5))
for network, label in LABELS.items():
    omega_eff_values = omega_eff(FREQS_HZ, baselines[network], H0_PER_S)
    amplitudes = power_law_amplitudes(FREQS_HZ, omega_eff_values, BETAS, REFERENCE_FREQUENCY_HZ, YEAR_S, 1.0)
    curve = pi_curve(FREQS_HZ, BETAS, amplitudes, REFERENCE_FREQUENCY_HZ)
    ax.loglog(FREQS_HZ, curve, label=label, ls="--" if network == "ET" else "-")
    print(f"{network:9s} PI minimum {curve.min():.2e} at {FREQS_HZ[np.argmin(curve)]:.1f} Hz", flush=True)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel(r"$\Omega_\mathrm{GW}(f)$")
ax.set_title(r"1$\sigma$ PI curves, T = 1 yr, 5-512 Hz", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "figures" / "pi_curves_et_designs.png"
fig.savefig(out, dpi=150)
print(f"saved {out}\n", flush=True)

networks = ("ET",) + DESIGN_NETWORKS[:3] + ("CE4020ET",) + DESIGN_NETWORKS[3:]
print("Fisher G mu threshold, data case 3:")
print(f"{'w':>4s}  " + " ".join(f"{network:>17s}" for network in networks), flush=True)
for label, w in EQUATIONS_OF_STATE.items():
    indices = [cosmic_string_tilt(w), FOREGROUNDS["CBC"][1], FOREGROUNDS["CBHE"][1]]
    thresholds = []
    for network in networks:
        sigma_cs = np.sqrt(amplitude_covariance(fisher_matrix(FREQS_HZ, baselines[network], indices, YEAR_S, H0_PER_S))[0, 0])
        thresholds.append(detection_threshold(sigma_cs, w, T_RD_GEV, PARAMS, TARGET, (1e-20, 1e-6)))
    print(f"{label:>4s}  " + " ".join(f"{threshold:17.2e}" for threshold in thresholds), flush=True)
