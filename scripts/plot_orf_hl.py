"""Plot the H1-L1 and H1-V1 tensor ORFs as a visual sanity check.

H1-L1 should start near -0.89 and oscillate toward zero above ~50 Hz.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sgwb_snr.orf import get_builtin_detector, tensor_orf

frequencies_hz = np.linspace(1.0, 500.0, 2000)
h1 = get_builtin_detector("H1")

fig, ax = plt.subplots(figsize=(7, 4))
for other in ["L1", "V1"]:
    gamma = tensor_orf(h1, get_builtin_detector(other), frequencies_hz)
    ax.plot(frequencies_hz, gamma, label=f"H1-{other}")
ax.axhline(0.0, color="grey", lw=0.5)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel(r"Tensor ORF $\gamma(f)$")
ax.set_title(r"Normalized: co-located, co-aligned detectors give $\gamma=1$", fontsize=10)
ax.set_xscale("log")
ax.legend()
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "figures" / "orf_h1_l1_v1.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"saved {out}")
