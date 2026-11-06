"""Table I (data case 3) of arXiv:2502.12828: our thresholds against the paper's.

G mu at which Delta(G mu)/(G mu) = 0.5, per network and equation of state w.
Fisher values are recomputed here (as in scripts/table1.py). The MCMC values are
the output of scripts/table1_mcmc.py (about 1 h to rerun), copied below; the
paper's values are its Table I.
"""

from pathlib import Path

import matplotlib.pyplot as plt
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
TARGET = 0.5
W_LABELS = ("1", "1/2", "1/3", "1/4")
W_VALUES = (1.0, 0.5, 1.0 / 3.0, 0.25)

# (CE4020, ET, CE4020ET) per w.
PAPER = {"1": (2.2e-16, 6.3e-17, 2.8e-17), "1/2": (8.9e-15, 2.5e-15, 5.6e-16),
         "1/3": (1.4e-13, 1.1e-14, 3.5e-15), "1/4": (1.4e-11, 3.5e-13, 1.3e-13)}
MCMC = {"1": (9.65e-16, 3.88e-16, 3.18e-16), "1/2": (2.40e-14, 5.41e-15, 3.23e-15),
        "1/3": (1.99e-13, 1.47e-14, 7.57e-15), "1/4": (1.24e-11, 2.16e-13, 9.31e-14)}

# Reference categorical palette, slots 1-3 (validated all-pairs, light mode);
# marker shape is a second, colour-independent encoding.
SERIES = [
    ("paper (MCMC, noisy simulation)", "#2a78d6", "o"),
    ("this work: MCMC, paper's priors", "#eb6834", "s"),
    ("this work: Fisher", "#1baf7a", "^"),
]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

baselines = {network: network_baselines(network, FREQS_HZ) for network in NETWORKS}
fisher_values = {}
for label, w in zip(W_LABELS, W_VALUES):
    indices = [cosmic_string_tilt(w), FOREGROUNDS["CBC"][1], FOREGROUNDS["CBHE"][1]]
    fisher_values[label] = tuple(
        detection_threshold(
            np.sqrt(amplitude_covariance(fisher_matrix(FREQS_HZ, baselines[n], indices, YEAR_S, H0_PER_S))[0, 0]),
            w, T_RD_GEV, PARAMS, TARGET, (1e-20, 1e-6),
        )
        for n in NETWORKS
    )

plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4), sharey=True, facecolor=SURFACE)
x = np.arange(len(W_LABELS))
for k, (ax, network) in enumerate(zip(axes, NETWORKS)):
    ax.set_facecolor(SURFACE)
    for (name, color, marker), table, dx in zip(SERIES, (PAPER, MCMC, fisher_values), (-0.15, 0.0, 0.15)):
        ax.semilogy(x + dx, [table[label][k] for label in W_LABELS], linestyle="none", marker=marker,
                    markersize=7, color=color, markeredgecolor=SURFACE, markeredgewidth=1.0,
                    label=name if k == 0 else None)
    ax.set_title(network, fontsize=10, color=INK)
    ax.set_xticks(x, W_LABELS)
    ax.set_xlim(-0.5, len(W_LABELS) - 0.5)
    ax.set_xlabel("equation of state w", color=MUTED)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
axes[0].set_ylabel(r"$G\mu$ where $\Delta G\mu / G\mu = 0.5$" + "\n(lower = more sensitive)")
fig.legend(loc="upper center", ncol=3, frameon=False, numpoints=1, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9))

out = Path(__file__).resolve().parents[1] / "figures" / "table1_comparison.png"
fig.savefig(out, dpi=170, facecolor=SURFACE)
print(f"saved {out}")
