# Noise curve sources

The CE and ET-D files are one-sided strain amplitude spectral densities (ASDs) in
1/sqrt(Hz); the ET-0304B-22 files are one-sided power spectral densities (PSDs) in
1/Hz. First column frequency in Hz. Files are unmodified.

| File | Detector | Source | Columns used |
|---|---|---|---|
| `cosmic_explorer_strain.txt` | CE-40 | Cosmic Explorer CE-T2000017, https://dcc.cosmicexplorer.org/CE-T2000017/public (zip dated 2025-10-27) | 0: f, 1: ASD |
| `cosmic_explorer_20km_strain.txt` | CE-20 | same archive | 0: f, 1: ASD |
| `ET-0000A-18_ETDSensitivityCurveTxtFile.txt` | ET-D | ET-Docs ET-0000A-18 (released 2018-02-15), https://apps.et-gw.eu/tds/ql/?c=12989 | 0: f, 3: ET-D-sum |
| `18213_ET10kmcolumns.txt` | ET 10 km (CoBA) | ET-Docs ET-0304B-22, "ET sensitivity curves used for CoBA Science Study" (Danilishin & Zhang, release B 2023-03-28), https://apps.et-gw.eu/tds/?content=3&r=18213 | 0: f, 3: LF+HF PSD |
| `18213_ET15kmcolumns.txt` | ET 15 km (CoBA) | same document | 0: f, 3: LF+HF PSD |

Downloaded: CE and ET-D 2026-11-03, ET-0304B-22 2026-11-04.

`CE-T2000017_README.txt` is the README shipped with the CE archive (CE strain
curves assume a source 15 deg off normal incidence).

The ET-D file has columns f, ET-D-LF, ET-D-HF, ET-D-sum. Following Hild et al.
(arXiv:1012.0908, Sec. 6), these describe one low/high-frequency pair of
interferometers with a 90-degree opening angle. The 60-degree geometry of the ET
triangle enters through the ORF factor sin(beta_1) sin(beta_2) in `orf.py`, not
through the PSD.

ET-0304B-22 (PDF, p. 1): columns are f, PSD of the high-frequency (HF)
interferometer, PSD of the low-frequency (LF) interferometer, PSD of the LF+HF
xylophone; each "corresponds to the strain sensitivity of a single L-shaped"
interferometer, so again no 60-degree factor belongs in the PSD. Verified:
column 3 = 1/(1/HF + 1/LF) to 1e-6, and the HF (LF) minimum lies at 270 (24) Hz
for 10 km. These are the curves used by Branchesi et al. (arXiv:2303.15923) for
the triangle vs 2L comparison (labelled "HFLF cryo" there).

SHA-256:

    ebc9145dc9079b9f8839730ba8ce6642dc25542b1fe63c22db90982abf61c29c  cosmic_explorer_strain.txt
    ced9a60e6a304a5809ee81ac24752d8fecf867c0a64f3056c915105a22869af2  cosmic_explorer_20km_strain.txt
    35ace6be81e56a56ed4f414a84e666aa5eeb99ec7cf3a875c5994981e26988a6  ET-0000A-18_ETDSensitivityCurveTxtFile.txt
    1970defcd160581a01690c6ce1bbb0fb91d778890b4e040caa560b72880bc49e  18213_ET10kmcolumns.txt
    839909b62a687243ebd8f8444239a98d22ec670f49d40360b9e68f0d9ead4fbb  18213_ET15kmcolumns.txt
