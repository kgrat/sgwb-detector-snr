# Noise curve sources

All files are one-sided strain amplitude spectral densities (ASDs) in 1/sqrt(Hz),
first column frequency in Hz. Downloaded 2026-11-03; files are unmodified.

| File | Detector | Source | Columns used |
|---|---|---|---|
| `cosmic_explorer_strain.txt` | CE-40 | Cosmic Explorer CE-T2000017, https://dcc.cosmicexplorer.org/CE-T2000017/public (zip dated 2025-10-27) | 0: f, 1: ASD |
| `cosmic_explorer_20km_strain.txt` | CE-20 | same archive | 0: f, 1: ASD |
| `ET-0000A-18_ETDSensitivityCurveTxtFile.txt` | ET-D | ET-Docs ET-0000A-18 (released 2018-02-15), https://apps.et-gw.eu/tds/ql/?c=12989 | 0: f, 3: ET-D-sum |

`CE-T2000017_README.txt` is the README shipped with the CE archive (CE strain
curves assume a source 15 deg off normal incidence).

The ET-D file has columns f, ET-D-LF, ET-D-HF, ET-D-sum. Following Hild et al.
(arXiv:1012.0908, Sec. 6), these describe one low/high-frequency pair of
interferometers with a 90-degree opening angle. The 60-degree geometry of the ET
triangle enters through the ORF factor sin(beta_1) sin(beta_2) in `orf.py`, not
through the PSD.

SHA-256:

    ebc9145dc9079b9f8839730ba8ce6642dc25542b1fe63c22db90982abf61c29c  cosmic_explorer_strain.txt
    ced9a60e6a304a5809ee81ac24752d8fecf867c0a64f3056c915105a22869af2  cosmic_explorer_20km_strain.txt
    35ace6be81e56a56ed4f414a84e666aa5eeb99ec7cf3a875c5994981e26988a6  ET-0000A-18_ETDSensitivityCurveTxtFile.txt
