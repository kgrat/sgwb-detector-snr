# sgwb-detector-snr

Python tools for computing the detectability (signal-to-noise ratio) of stochastic
gravitational-wave backgrounds (SGWB) across ground-based detector networks:
LIGO, Einstein Telescope (ET), and Cosmic Explorer (CE).

Developed with Claude Code (Anthropic) as a coding assistant; all physical
choices, conventions and validations were reviewed and decided by me.

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
| 7 | Fisher component separation, cross-checked by MCMC with the paper's priors | Gaussian limit of the MCMC equals Fisher; Table I, w = 1/3 row (MCMC: ET 1.3x, CE4020 1.4x, CE4020ET 2.2x the paper's G mu) | Done |
| 8 | Data case 2: CBHE in the data but not in the model | Unmodelled copy of a modelled shape is absorbed exactly; paper Sec. V trend (weak for CE4020, strong for ET and CE4020ET) | Done |
| 9 | Extension: ET triangle vs 2L (Branchesi et al., arXiv:2303.15923) | Parallel-arm angle 2.51 deg and gamma(f -> 0) = 0 at 47.51 deg; 2L-0 beats the triangle below ~100 Hz (we find 115 Hz) | Done |
| 10 | Extension: correlated Newtonian noise in the ET triangle (Caporali et al., arXiv:2501.09057) | Uncorrelated limit equals the paper's Fisher matrix; Isserlis covariance; sign of the bias | Done (Fisher) |

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
    python scripts/table1.py
    python scripts/table1_mcmc.py   # ~1 h on one core
    python scripts/table1_case2.py  # ~1 h; one w per process: table1_case2.py 1/3
    python scripts/et_designs.py
    python scripts/correlated_noise.py

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
- CE and ET geometry from Gupta et al. (arXiv:2307.10421), Table II: latitude,
  longitude and x-arm azimuth measured counter-clockwise from local east; the
  y-arm is 90 deg (CE) or 60 deg (ET) further counter-clockwise; elevation 0 m
  (not given in the table).
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
- Component separation: Fisher matrix of the three amplitudes (paper Eqs. 30-31,
  continuum limit, same 2T normalization as the SNR), marginalized over the
  foregrounds; Delta(G mu)/(G mu) = sigma_CS / (k Omega_CS(f*)) with
  k = d ln Omega_CS / d ln(G mu) (k = 1/2 on the w = 1/3 plateau). This is the
  Gaussian, prior-free limit of the paper's MCMC (Eq. 37).
- MCMC cross-check (`src/sgwb_snr/mcmc.py`): the paper's Gaussian likelihood
  (Eq. 37) in the three amplitudes, with the Fisher covariance and log-uniform
  priors on G mu, Omega_CBC and Omega_CBHE in [1e-18, 1e-8] (paper Sec. V),
  sampled with emcee. Omega_hat is the injected signal (a noise-free forecast);
  the paper used one simulated noisy realization. Chains run until they are
  longer than 50 autocorrelation times and tau has settled to 1%;
  5 tau are discarded. Delta(G mu)/(G mu) is half the 16-84 percentile width of
  G mu over its median (our choice: the paper does not define it for a
  non-Gaussian posterior). The threshold is found by scanning 12 values of G mu
  from 1/100 to 2 times the Fisher threshold and interpolating in log-log.
- Data case 2 (`scripts/table1_case2.py`): the data hold CS + CBC + CBHE, the
  model only CS + CBC (paper Sec. IV B). The noise-free estimate (paper Eq. 26)
  of the modelled amplitudes is shifted by (F_mm)^-1 F_mu Omega_CBHE
  (`fisher.unmodelled_bias`); the MCMC then uses the CS + CBC block of the
  covariance. Because the estimate is biased, Delta(G mu)/(G mu) is the total
  error with respect to the injected value,
  sqrt((median - truth)^2 + halfwidth^2) / truth, because the posterior width
  alone is blind to bias; the case-3 measure is also printed.
  G mu is scanned in 0.25 dex steps up to 1e-10, where the paper stops ("> 1e-10").
- ET designs (`scripts/et_designs.py`): Branchesi et al. (arXiv:2303.15923) 2L at
  Sardinia (40 31', 9 25') and Meuse-Rhine (50 43' 23", 5 55' 14"), 15 km arms;
  Sardinia arms east/north, the Meuse-Rhine L rotated counter-clockwise by
  alpha = 0 or 45 deg (their App. B). Noise: ET-0304B-22 (CoBA) LF+HF curves,
  10 km for the triangle and 15 km for the 2L, as in Branchesi et al.
- Correlated noise (`src/sgwb_snr/correlated.py`, `scripts/correlated_noise.py`):
  Caporali et al. (arXiv:2501.09057) Eq. 11, CSD N_o = r N_d(2.75 Hz)
  (f / 2.75 Hz)^-8 (body-wave Newtonian noise) between every pair of the
  triangle, r in {-0.4, ..., 0.8}. The cross-spectra of all baselines get the
  exact Gaussian covariance N_IK N_JL + N_IL N_JK, which reduces to the
  paper's Eq. 25 for uncorrelated noise; two baselines sharing a detector are
  then correlated at first order in N_o. "Modelled": r is a free amplitude
  (tilt fixed) and the covariance includes N_o. "Unmodelled": the analyst uses
  the uncorrelated covariance and no r, so the estimate is biased;
  Delta(G mu)/(G mu) = sqrt(sigma^2 + bias^2) / (k Omega_CS), the Fisher form of
  the data case 2 measure. Fisher only.

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

Table I (data case 3: CS, CBC and CBHE all modelled), G mu at which
Delta(G mu)/(G mu) = 0.5. Ours by Fisher and by MCMC (paper's priors) / paper
(MCMC); the ratio is MCMC / paper:

| w | CE4020 | ET | CE4020ET |
|---|---|---|---|
| 1 | 1.67e-15, 9.65e-16 / 2.2e-16 (4.4x) | 2.88e-16, 3.88e-16 / 6.3e-17 (6.2x) | 2.16e-16, 3.18e-16 / 2.8e-17 (11x) |
| 1/2 | 3.75e-14, 2.40e-14 / 8.9e-15 (2.7x) | 3.03e-15, 5.41e-15 / 2.5e-15 (2.2x) | 1.94e-15, 3.23e-15 / 5.6e-16 (5.8x) |
| 1/3 | 3.91e-13, 1.99e-13 / 1.4e-13 (1.4x) | 1.02e-14, 1.47e-14 / 1.1e-14 (1.3x) | 5.55e-15, 7.57e-15 / 3.5e-15 (2.2x) |
| 1/4 | 3.20e-11, 1.24e-11 / 1.4e-11 (0.89x) | 1.65e-13, 2.16e-13 / 3.5e-13 (0.62x) | 7.17e-14, 9.31e-14 / 1.3e-13 (0.72x) |

The ordering of networks and of w matches the paper. The CS-CBC amplitude
correlation is about -0.98 in every network, and marginalizing over the
foregrounds inflates sigma_CS by 5x (ET), 9.5x (CE4020) and up to 29x (w = 1),
so the result is sensitive to how the degeneracy is handled.

The priors move the thresholds in opposite directions depending on the network:

- CE4020: MCMC is 1.6-2.6x lower than Fisher, i.e. closer to the paper. The
  injected Omega_CBHE is only 0.44 sigma for CE4020, so the log-uniform prior
  (which forbids negative amplitudes) cuts off half of the degenerate
  direction that Fisher marginalizes over.
- ET and CE4020ET: MCMC is 1.3-1.8x higher than Fisher. Here Omega_CBHE is
  measured at about 4 sigma, so positivity does little; instead, at
  Delta ~ 0.5 the posterior on G mu is skewed by the log-uniform prior toward
  the 1e-18 floor, and half the 16-84% width over the median exceeds the
  linear-propagation value.

Including the priors brings the w = 1/3 and 1/4 rows within 0.6-2.2x of the
paper. They do not explain the w = 1 and 1/2 rows, where we remain 2-11x weaker.
Remaining known differences: the ET noise and ORF (above), the paper's single
noisy realization of Omega_hat, and the unstated definition of Delta(G mu)/(G mu)
for a non-Gaussian posterior.

Table I, data case 2 (CBHE not modelled), MCMC, error with respect to the truth
/ paper (ratio):

| w | CE4020 | ET | CE4020ET |
|---|---|---|---|
| 1 | 7.06e-16 / 6.3e-16 (1.1x) | 6.16e-16 / 2.0e-15 (0.31x) | 5.20e-16 / 5.6e-15 (0.093x) |
| 1/2 | 2.16e-14 / 5.6e-14 (0.39x) | 1.11e-14 / 3.2e-14 (0.35x) | 7.80e-15 / 2.0e-14 (0.39x) |
| 1/3 | 1.71e-13 / 5.6e-13 (0.31x) | 5.43e-14 / 2.0e-13 (0.27x) | 3.26e-14 / 5.0e-14 (0.65x) |
| 1/4 | 9.16e-12 / > 1e-10 | 1.35e-12 / > 1e-10 | 6.75e-13 / > 1e-10 |

The Fisher bias from the unmodelled CBHE explains the paper's qualitative
finding: it is 0.6-0.7 sigma for CE4020 but 3-5 sigma for ET and CE4020ET. For
w <= 1/2 it is negative (CBC absorbs the CBHE power and overshoots, and the
anti-correlated CS amplitude is pulled down); for w = 1 it is positive (the CS
tilt +1 lies between the CBC and CBHE tilts, so CS impersonates part of the
CBHE). Relative to our data case 3, ignoring CBHE degrades ET and CE4020ET by
1.6x (w = 1) to 6-7x (w = 1/4), and leaves CE4020 unchanged or slightly better
(a two-component fit has a smaller error and its bias is small).

The paper finds much larger degradations (13x to more than 700x) and no
detection below 1e-10 at w = 1/4, which a noise-free Fisher bias does not
produce: at G mu = 1e-12 the w = 1/4 CS amplitude, 1.4e-11 at 25 Hz, is
already about 5x the ~3e-12 bias, and it keeps growing with G mu. The paper's single noisy realization (whose data case 1 already
shows spurious CS amplitudes of 3.5e-13 to 1.1e-11, its Fig. 6) is the most
likely source; we cannot test this without their simulated data. With the
case-3 measure (posterior width over median) the positive w = 1 bias would make
ET and CE4020ET look detectable already at G mu < 3e-17: the posterior then
confidently measures leaked CBHE power, which is why we report the error with
respect to the truth.

## Extensions

### ET triangle vs 2L

1-sigma PI curves (`figures/pi_curves_et_designs.png`, 1 yr, 5-512 Hz): ET-D
triangle 4.5e-13, CoBA 10 km triangle 7.9e-13, 2L-15km at alpha = 0 deg 2.8e-13,
at 45 deg 3.2e-12 (minima, all near 8-9 Hz). As in Branchesi et al. (Fig. 27
and Sec. 8.1.1), the parallel 2L is the most sensitive below 115 Hz and the
triangle above; the 45-degree 2L is 10x worse than the parallel one at 10-50 Hz
and comparable (1.4x) at 300 Hz. The best and worst orientations for stochastic
searches are alpha = 2.51 and 47.51 deg (great-circle parallel and 45 deg
from it), which we reproduce from the site coordinates alone.

Data case 3, Fisher G mu at Delta(G mu)/(G mu) = 0.5 (ET-D columns as in Table I):

| w | ET-D tri | CoBA tri 10 km | 2L 0 deg | 2L 45 deg | CE4020ET | CE4020 + CoBA tri | CE4020 + 2L 0 | CE4020 + 2L 45 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.88e-16 | 3.06e-16 | 1.90e-16 | 1.73e-15 | 2.16e-16 | 2.54e-16 | 1.69e-16 | 5.08e-16 |
| 1/2 | 3.03e-15 | 3.77e-15 | 1.41e-15 | 3.77e-14 | 1.94e-15 | 2.74e-15 | 1.19e-15 | 5.84e-15 |
| 1/3 | 1.02e-14 | 1.57e-14 | 3.27e-15 | 3.49e-13 | 5.55e-15 | 9.80e-15 | 2.58e-15 | 2.46e-14 |
| 1/4 | 1.65e-13 | 3.35e-13 | 3.20e-14 | 2.33e-11 | 7.17e-14 | 1.72e-13 | 2.32e-14 | 5.64e-13 |

For cosmic strings the parallel 2L is the best ET design: against the CoBA
triangle it improves the threshold by 1.6x (w = 1) to 10x (w = 1/4), alone, and
by 1.5-7x with CE4020. The CS signal and its separation from the foregrounds
are decided below ~30 Hz, where the parallel 2L is more sensitive; the
improvement grows as w falls because the CS spectrum becomes redder. The
45-degree 2L is the worst option, 9x (w = 1) to 730x (w = 1/4) worse than the
parallel 2L alone; with CE4020 the loss shrinks to 3-24x.

### Correlated Newtonian noise in the triangle

Per unit r, N_o / N_d is 0.63 (ET-D) and 0.41 (CoBA 10 km) at 5 Hz and falls
below 1e-2 above 10 Hz, so all of Caporali et al.'s r values keep the noise
matrix positive definite. But in Omega units the correlated noise at 5 Hz is of
order 1e-7, four orders of magnitude above the cosmic-string signal.

Fisher G mu threshold, w = 1/3 (`scripts/correlated_noise.py` prints all w):

| r | ET mod. | ET unmod. | CE4020ET mod. | CE4020ET unmod. | CoBA tri mod. | CE4020 + CoBA tri mod. |
|---|---|---|---|---|---|---|
| none (r = 0, not fitted) | 1.02e-14 | | 5.55e-15 | | 1.57e-14 | 9.80e-15 |
| 0 | 2.21e-14 | 1.02e-14 | 1.03e-14 | 5.55e-15 | 2.90e-14 | 1.52e-14 |
| +-0.2 | 2.11-2.31e-14 | > 1e-10 | 1.00-1.05e-14 | > 1e-10 | 2.85-2.96e-14 | 1.51-1.53e-14 |
| +-0.4 | 2.04-2.42e-14 | > 1e-10 | 0.98-1.07e-14 | > 1e-10 | 2.80-3.02e-14 | 1.50-1.54e-14 |
| 0.8 | 2.64e-14 | > 1e-10 | 1.09e-14 | > 1e-10 | 3.15e-14 | 1.56e-14 |

- Unmodelled, any |r| >= 0.2 makes the CS background unmeasurable below
  G mu = 1e-10 for w <= 1/3; at w = 1/2 the threshold moves from ~3e-15 to
  2e-11 - >1e-10, and at w = 1 from ~3e-16 to 1e-13 - 1e-12. The bias is linear in r, so +r and -r
  give the same threshold.
- Modelled (one extra free amplitude, tilt -8), the cost is nearly independent
  of r: 1.2x (w = 1), 1.5x (w = 1/2), 2x (w = 1/3) and 3.5x (w = 1/4) for ET
  alone, 1.2-2.7x for CE4020ET. Most of it is the marginalization itself: the
  correlated noise and the redder CS spectrum compete for the same 5-10 Hz.
- With the noise modelled, the parallel 2L (no co-located pairs, so no such
  noise) beats the CoBA triangle at w = 1/3 by 9x (alone) and 6x (with CE4020).

Limits: power-law NN template with its tilt fixed and the same CSD for all
three pairs (Caporali et al.); correlated magnetic (Schumann) noise, which
would also couple the two 2L sites, is not included; Fisher only.
