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
| 4 | SNR for an arbitrary Omega_GW(f) (Thrane & Romano 2013, Eq. 22) | Analytic constant-noise integral; T&R Eq. 26 for co-located detectors | Done |
| 5 | Power-law integrated curves (Thrane & Romano 2013) | Flat spectrum on the PI curve gives SNR = 1; paper Fig. 1, right panel (CE4020 within 0.84-1.10; ET differs) | Done |
| 6 | Cosmic-string, CBC, and hyperbolic-encounter spectra | Omega_CS proportional to sqrt(G mu) on the plateau; Figs. 2-3 (within 6%) | Done |
| 7 | Fisher component separation | Table I, w = 1/3 row | To do |

## Setup

    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
    pytest
    python scripts/plot_orf_hl.py
    python scripts/plot_orf_next_generation.py
    python scripts/plot_asd.py
    python scripts/plot_pi_curves.py
    python scripts/plot_spectra.py

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
- SNR (Thrane & Romano 2013, Eq. 22) with Gamma = gamma / 5:
  rho^2 = 2T sum_{I<J} int df gamma^2 (3 H0^2 / 10 pi^2)^2 Omega^2 / (f^6 P_I P_J),
  the same normalization as the paper's Fisher matrix (Eqs. 30-31).
  H0 = 67.8 km/s/Mpc (paper, Sec. III). Noise is assumed uncorrelated between
  all detectors, including the co-located ET interferometers.
- PI curves (Thrane & Romano Eqs. 29-30): 1 sigma, T = 1 yr, integrated over the
  paper's analysis band 5-512 Hz at 1/8 Hz, envelope over beta = -8 to 8 in
  steps of 0.01 (integer-only beta sags by up to 25% between tangent points).
- Cosmic strings: the paper's Eqs. 3-6 (Sousa et al. 2020 template) with its
  parameters, reading "alpha = 0.1" as loops born at l = 0.1 t
  (eps_r = 0.1 / (Gamma G mu)); see "Known differences". Non-standard
  cosmologies follow the paper's Eqs. 7-9 (pivot at f_rd, T_rd = 1 GeV), for
  1/9 < w <= 1. CBC and CBHE are the paper's power laws (Eqs. 12, 14).

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

Fig. 1 (right), 1 sigma PI curves, ratio ours / paper (figure read-off):

| f [Hz] | CE4020 | ET | CE4020ET |
|---|---|---|---|
| 6 | 1.09 | 0.39 | 0.46 |
| 10 | 0.88 | 0.47 | 0.54 |
| 30 | 0.84 | 0.70 | 0.84 |
| 100 | 0.89 | 0.82 | 0.88 |
| 300 | 1.10 | - | 0.96 |

CE4020's minimum matches (1.6e-12 near 14 Hz in both). ET is up to 2.5x more
sensitive at low frequency, which follows from the ET-D noise difference above
(at 6 Hz our PSD is 0.52x the paper's). The paper's curves continue to 2 kHz and
are only reproduced above ~300 Hz if the integration also extends to 2 kHz
(CE4020 then within 0.82-1.09 everywhere), although the paper states a
5-512 Hz analysis band; we keep 5-512 Hz.

Cosmic-string amplitude. Evaluated as printed, the paper's Eq. 3
(eps_r = alpha xi_r / (Gamma G mu), alpha = 0.1, xi_r = 0.271) gives a w = 1/3
plateau of 5.14e-10 at G mu = 1e-10, but the paper's Figs. 2-3 show 9.92e-10.
Loops born at l = 0.1 t (eps_r = 0.1 / (Gamma G mu)) give 9.87e-10, and this is
the l = 0.1 t, F = 0.1 calibration behind the paper's A_r = 0.54 (Sousa et al.,
Sec. V), so we use it. The printed reading would weaken every G mu limit by
1.93^2 = 3.7. With this choice our spectra match Fig. 2 to within 6%
(G mu = 1e-10 curves within 1.5%; the paper's G mu = 1e-16 curves are ~4% higher
than ours).
