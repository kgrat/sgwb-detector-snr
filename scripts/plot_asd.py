"""Plot CE-40, CE-20 and ET-D strain ASDs, as in Fig. 1 (left) of arXiv:2502.12828.

Official files (data/noise_curves/SOURCES.md). Above 100 Hz they agree with the
paper within 10%; below ~60 Hz ET-D differs in shape (see README,
"Known differences"). The shaded band is the paper's analysis band, 5-512 Hz.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sgwb_snr.noise import psd

frequencies_hz = np.logspace(np.log10(5.0), np.log10(2000.0), 3000)  # axis range of Fig. 1

fig, ax = plt.subplots(figsize=(6, 4.5))
for detector in ["ET-D", "CE-40", "CE-20"]:
    ax.loglog(frequencies_hz, np.sqrt(psd(detector, frequencies_hz)), label=detector)
ax.axvspan(5.0, 512.0, color="grey", alpha=0.1, label="analysis band")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel(r"Strain ASD [1/$\sqrt{\mathrm{Hz}}$]")
ax.set_title("ET-D is the 90-degree-pair curve; 60-degree factor is in the ORF", fontsize=9)
ax.legend()
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "figures" / "asd_next_generation.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"saved {out}")
