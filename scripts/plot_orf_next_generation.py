"""Plot CE/ET tensor ORFs for comparison with Fig. 1 (middle) of arXiv:2502.12828.

Geometry from Gupta et al. (arXiv:2307.10421) Table II. CE-40 - CE-20 should
start at +0.134 as in the paper. Our ET curves differ from the paper's (see
README, "Known differences"): each ET pair starts at -3/8, and the three
ETk - CE curves sum to zero at low frequency.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sgwb_snr.detectors import cosmic_explorer, einstein_telescope
from sgwb_snr.orf import tensor_orf

BAND_HZ = (5.0, 512.0)  # analysis band of arXiv:2502.12828
frequencies_hz = np.logspace(0.0, np.log10(2000.0), 3000)  # axis range of Fig. 1

ce40 = cosmic_explorer("CE-A", 40.0, *BAND_HZ)
ce20 = cosmic_explorer("CE-B", 20.0, *BAND_HZ)
et = einstein_telescope(*BAND_HZ)

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

ax_left.plot(frequencies_hz, tensor_orf(ce40, ce20, frequencies_hz), label="CE-40 - CE-20")
ax_left.plot(frequencies_hz, tensor_orf(et[0], et[1], frequencies_hz), label="ET1 - ET2")
ax_left.set_title("Matches paper (CE); ET pair = -3/8", fontsize=10)

for k, ifo in enumerate(et):
    ax_right.plot(frequencies_hz, tensor_orf(ifo, ce40, frequencies_hz), color=f"C{k}", label=f"{ifo.name} - CE-40")
    ax_right.plot(frequencies_hz, tensor_orf(ifo, ce20, frequencies_hz), color=f"C{k}", ls="--", label=f"{ifo.name} - CE-20")
ax_right.set_title("ET interferometers with CE", fontsize=10)

for ax in (ax_left, ax_right):
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.legend(fontsize=8)
ax_left.set_ylabel(r"Tensor ORF $\gamma(f)$")
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "figures" / "orf_next_generation.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"saved {out}")
