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
| 2 | CE-40 / CE-20 / ET-D geometry (Gupta et al., arXiv:2307.10421, Table II) | Paper Fig. 1, middle panel | To do |
| 3 | Detector ASDs | Paper Fig. 1, left panel | To do |
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
