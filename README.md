# sgwb-detector-snr

Python tools for computing the detectability (signal-to-noise ratio) of stochastic
gravitational-wave backgrounds (SGWB) across ground-based detector networks:
LIGO, Einstein Telescope (ET), and Cosmic Explorer (CE).

## Current goal

Reproduce and extend Wang, Jin & Zhang, *Detecting SGWB from cosmic strings with
next-generation detector networks* ([arXiv:2502.12828](https://arxiv.org/abs/2502.12828)):
power-law integrated sensitivity curves for CE/ET networks, then Fisher-matrix
component separation of a cosmic-string background from CBC and hyperbolic-encounter
foregrounds.

## Status

| Step | What | Validation target | Status |
|---|---|---|---|
| 1 | Tensor overlap reduction functions | H1-L1 gamma(0) = -0.89; co-located = 1 | Done |
| 2 | CE-40 / CE-20 / ET-D geometry (Gupta et al., arXiv:2307.10421, Table II) | Paper Fig. 1, middle panel (CE matches; ET differs, see below) | Done |
| 3 | Detector ASDs (official CE-T2000017 and ET-0000A-18 files) | Paper Fig. 1, left panel (within 10% above 100 Hz; ET-D differs below ~60 Hz) | Done |
| 4 | SNR for an arbitrary Omega_GW(f) | Flat spectrum on the PI curve gives SNR = 1 | To do |
| 5 | Power-law integrated curves (Thrane & Romano 2013) | Paper Fig. 1, right panel | To do |
| 6 | Cosmic-string, CBC, and hyperbolic-encounter spectra | Omega_CS proportional to sqrt(G mu) on the plateau; Figs. 2-3 | To do |
| 7 | Fisher component separation | Table I, w = 1/3 row | To do |

## Setup

    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
    pytest
    python scripts/plot_orf_hl.py
    python scripts/plot_orf_next_generation.py
    python scripts/plot_asd.py

Versions are pinned exactly. gwpy and matplotlib are pinned below their latest
releases because newer versions break pygwb 1.5.1 (see comments in requirements.txt).

On Apple Silicon macOS, two transitive dependencies (`ligo-segments`, via gwpy, and
`python-ligo-lw`, via pygwb) have no prebuilt wheels and are compiled from source.
Recent Apple clang treats `-Wint-conversion` as an error and the build fails. If
`pip install -r requirements.txt` stops there, rerun it as:

    CFLAGS="-Wno-error=int-conversion" pip install -r requirements.txt

This only downgrades that one diagnostic to a warning; no pinned version changes.

## Conventions

- Frequencies in Hz throughout.
- Tensor ORF normalized so co-located, co-aligned L-shaped detectors give gamma = 1.
- Next-generation detectors are never taken from bilby's built-in files: bilby's
  CE detector sits at the Hanford site. They are defined explicitly from the
  published configurations.
- Detectors with non-90-degree arms (the three 60-degree ET interferometers) get
  the factor sin(beta_1) sin(beta_2) on top of pygwb's ORF, which assumes
  L-shaped detectors. Two ET interferometers give gamma(f -> 0) = -3/8.
- ET is built as three explicit interferometers at the triangle's vertices.
  bilby's `TriangularInterferometer` is not used: it swaps the arm pairs of ET2
  and ET3.
- Noise curves are the official files in `data/noise_curves/` (provenance in
  `SOURCES.md`). The ET-D curve describes a 90-degree interferometer pair
  (Hild et al., arXiv:1012.0908, Sec. 6), so it is used as-is; the 60-degree
  geometry enters only through the ORF factor above.

## Known differences from the paper

Fig. 1 (middle) of arXiv:2502.12828, read off the figure at 1 Hz, compared with
this code using Gupta et al. Table II geometry:

| Curve | Paper | This code |
|---|---|---|
| CE-40 - CE-20 | +0.134 | +0.134 |
| ET-D self | -0.384, flat to 2 kHz | -0.375 for each ET pair |
| ET-D - CE-40 | -0.316 | -0.059 / -0.230 / +0.289 (ET1 / ET2 / ET3) |
| ET-D - CE-20 | -0.139 | -0.231 / -0.098 / +0.329 (ET1 / ET2 / ET3) |

No single interferometer at the Table II ET site, at any x-arm orientation, with
or without the opening-angle factor, reproduces both paper ET-CE values (best
mismatch 0.024), and bilby's built-in ET does not either. The paper's ET-D self
value is also not the -3/8 that two coplanar 60-degree interferometers rotated
by 120 degrees must give. We therefore keep the physically derived ET geometry and expect the ET and
CE4020ET results to differ from the paper's.

Fig. 1 (left) of arXiv:2502.12828 does not name its noise files. Ratio of the
official files to the paper's curves (figure read-off):

| f [Hz] | ET-D | CE-40 | CE-20 |
|---|---|---|---|
| 6 | 0.72 | 1.09 | 1.02 |
| 30 | 1.27 | 0.94 | 0.96 |
| 100 | 1.05 | 0.96 | 0.97 |
| 300 | 1.04 | 0.93 | 0.96 |
| 1000 | 1.03 | 0.92 | 0.92 |

The CE shapes match but our curves are 4-8% lower, consistent with the paper
using an earlier CE-T2000017 release (ours is dated 2025-10-27). The paper's
ET-D agrees above ~100 Hz but has a different shape below ~60 Hz (no bump near
30 Hz, different line features), so it is not the ET-0000A-18 file the paper
cites.
