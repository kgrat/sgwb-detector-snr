"""Source spectra, as in Figs. 2 and 3 of arXiv:2502.12828.

Left (Fig. 2): cosmic strings for G mu = 1e-10 (solid) and 1e-16 (dashed),
w = 1, 1/2, 1/3, 1/4, T_rd = 1 GeV. Right (Fig. 3): G mu = 1e-10, w = 1/3 cosmic
strings, the CBC and CBHE foregrounds and their sum, over our 1 sigma PI
curves (1 yr, 5-512 Hz).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sgwb_snr.networks import NETWORKS, network_baselines
from sgwb_snr.pi_curve import pi_curve, power_law_amplitudes
from sgwb_snr.snr import hubble_constant_si, omega_eff
from sgwb_snr.spectra import WANG_JIN_ZHANG_2025 as PARAMS
from sgwb_snr.spectra import foreground, omega_cosmic_strings

H0_PER_S = hubble_constant_si(PARAMS.h)
YEAR_S = 365.25 * 86400.0
BAND_HZ = np.arange(5.0, 512.0 + 1e-9, 0.125)
BETAS = np.round(np.arange(-8.0, 8.0 + 1e-9, 0.01), 2)
T_RD_GEV = 1.0  # paper Sec. III A 2

fig, (ax_w, ax_total) = plt.subplots(1, 2, figsize=(12, 4.8))

wide_hz = np.logspace(-3.0, 4.0, 2000)
for k, (w, label) in enumerate([(1.0, "1"), (0.5, "1/2"), (1.0 / 3.0, "1/3"), (0.25, "1/4")]):
    for string_tension, style in [(1e-10, "-"), (1e-16, "--")]:
        ax_w.loglog(
            wide_hz, omega_cosmic_strings(wide_hz, string_tension, w, T_RD_GEV, PARAMS),
            color=f"C{k}", ls=style, label=rf"$G\mu$={string_tension:.0e}, w={label}",
        )
ax_w.set_ylim(1e-13, 1e-6)
ax_w.set_title("Cosmic strings (cf. paper Fig. 2)", fontsize=10)
ax_w.legend(fontsize=7, ncol=2)

# Most sensitive network first, so each less sensitive region is drawn on top.
for network, colour in [("CE4020ET", "lightsteelblue"), ("ET", "papayawhip"), ("CE4020", "lightgrey")]:
    assert network in NETWORKS
    omega_eff_values = omega_eff(BAND_HZ, network_baselines(network, BAND_HZ), H0_PER_S)
    amplitudes = power_law_amplitudes(BAND_HZ, omega_eff_values, BETAS, 25.0, YEAR_S, 1.0)
    ax_total.fill_between(BAND_HZ, pi_curve(BAND_HZ, BETAS, amplitudes, 25.0), 1e-5, color=colour, label=f"1$\\sigma$ PI ({network})")
cosmic = omega_cosmic_strings(BAND_HZ, 1e-10, 1.0 / 3.0, T_RD_GEV, PARAMS)
cbc, cbhe = foreground("CBC", BAND_HZ), foreground("CBHE", BAND_HZ)
ax_total.loglog(BAND_HZ, cosmic, label=r"CS ($G\mu$=1e-10, w=1/3)")
ax_total.loglog(BAND_HZ, cbc, label="CBC")
ax_total.loglog(BAND_HZ, cbhe, label="CBHE")
ax_total.loglog(BAND_HZ, cosmic + cbc + cbhe, color="k", label="total")
ax_total.set_ylim(1e-13, 1e-5)
ax_total.set_title("Components, 5-512 Hz (cf. paper Fig. 3)", fontsize=10)
ax_total.legend(fontsize=7)

for ax in (ax_w, ax_total):
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$\Omega_\mathrm{GW}(f)$")
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "figures" / "spectra.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"saved {out}")
