"""1-sigma power-law integrated curves for CE4020, ET and CE4020ET, 1 yr.

Compare with Fig. 1 (right) of arXiv:2502.12828. Integration band 5-512 Hz at
1/8 Hz resolution (paper, Secs. II and IV B); beta from -8 to 8 in steps of
0.01 (a smooth envelope, unlike integer-only beta). The
paper's figure appears to be integrated up to 2 kHz, so it differs above
~300 Hz; see README, "Known differences".
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sgwb_snr.networks import NETWORKS, network_baselines
from sgwb_snr.pi_curve import pi_curve, power_law_amplitudes
from sgwb_snr.snr import hubble_constant_si, omega_eff

H0_PER_S = hubble_constant_si(0.678)  # arXiv:2502.12828 Sec. III
YEAR_S = 365.25 * 86400.0
FREQS_HZ = np.arange(5.0, 512.0 + 1e-9, 0.125)
BETAS = np.round(np.arange(-8.0, 8.0 + 1e-9, 0.01), 2)
F_REF_HZ = 25.0  # the paper's reference frequency f*

fig, ax = plt.subplots(figsize=(6, 4.5))
for network in NETWORKS:
    omega_eff_values = omega_eff(FREQS_HZ, network_baselines(network, FREQS_HZ), H0_PER_S)
    amplitudes = power_law_amplitudes(FREQS_HZ, omega_eff_values, BETAS, F_REF_HZ, YEAR_S, 1.0)
    curve = pi_curve(FREQS_HZ, BETAS, amplitudes, F_REF_HZ)
    ax.loglog(FREQS_HZ, curve, label=rf"1$\sigma$ PI ({network})")
    print(f"{network:9s} minimum {curve.min():.2e} at {FREQS_HZ[np.argmin(curve)]:.1f} Hz")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel(r"$\Omega_\mathrm{GW}(f)$")
ax.set_title("T = 1 yr, 5-512 Hz", fontsize=10)
ax.legend()
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "figures" / "pi_curves.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"saved {out}")
