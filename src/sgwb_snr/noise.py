"""Detector noise power spectral densities (PSDs) from official sensitivity files.

Files and provenance: data/noise_curves/SOURCES.md. The CE and ET-D files hold
one-sided strain ASDs (PSD = ASD^2); the ET-0304B-22 files (CoBA curves used by
Branchesi et al., arXiv:2303.15923) hold one-sided PSDs, column 3 = LF+HF.

ET-D: the ET-0000A-18 curve describes one low/high-frequency pair of
interferometers with a 90-degree opening angle (Hild et al., arXiv:1012.0908,
Sec. 6). The 60-degree triangle geometry is applied once, in the ORF
(sin(beta_1) sin(beta_2) in orf.py), so it must NOT be applied here as well.
The same holds for the CoBA curves: ET-0304B-22 states that each is the strain
sensitivity of a single L-shaped interferometer.
"""

from pathlib import Path

import numpy as np

_NOISE_DIR = Path(__file__).resolve().parents[2] / "data" / "noise_curves"

# detector name -> (file name, column, what the column holds: "asd" or "psd")
_NOISE_FILES = {
    "CE-40": ("cosmic_explorer_strain.txt", 1, "asd"),
    "CE-20": ("cosmic_explorer_20km_strain.txt", 1, "asd"),
    "ET-D": ("ET-0000A-18_ETDSensitivityCurveTxtFile.txt", 3, "asd"),  # ET-D-sum
    # ET-0304B-22 columns: f, HF, LF, LF+HF (checked: col 3 = 1/(1/HF + 1/LF)).
    "ET-10km-CoBA": ("18213_ET10kmcolumns.txt", 3, "psd"),
    "ET-15km-CoBA": ("18213_ET15kmcolumns.txt", 3, "psd"),
}


def load_asd(detector):
    """Tabulated (frequencies_hz, asd_per_sqrt_hz) from the source file (sqrt taken for PSD files)."""
    if detector not in _NOISE_FILES:
        raise ValueError(f"No noise curve for '{detector}'. Available: {sorted(_NOISE_FILES)}")
    file_name, column, quantity = _NOISE_FILES[detector]
    table = np.loadtxt(_NOISE_DIR / file_name)
    assert table.ndim == 2 and table.shape[1] > column, f"unexpected table shape {table.shape}"
    frequencies_hz, values = table[:, 0], table[:, column]
    if np.any(np.diff(frequencies_hz) <= 0):
        raise ValueError(f"{file_name}: frequencies not strictly increasing")
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError(f"{file_name}: {quantity.upper()} must be finite and positive")
    asd_per_sqrt_hz = {"asd": values, "psd": np.sqrt(values)}[quantity]
    return frequencies_hz, asd_per_sqrt_hz


def psd(detector, frequencies_hz):
    """One-sided strain PSD [1/Hz] of `detector` at `frequencies_hz`.

    Interpolates log(ASD) linearly in log(f), i.e. a power law between
    adjacent tabulated points (the files have 1000-3000 log-spaced samples).
    Raises for frequencies outside the tabulated range: no extrapolation.
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    assert frequencies_hz.ndim == 1, "frequencies_hz must be 1D"
    table_hz, table_asd = load_asd(detector)
    # Tolerate floating-point rounding of the tabulated end points (the CE files
    # start at 5.000000000000000888 Hz), but never real extrapolation.
    rounding = 1e-9
    if frequencies_hz.min() < table_hz[0] * (1 - rounding) or frequencies_hz.max() > table_hz[-1] * (1 + rounding):
        raise ValueError(
            f"{detector} noise curve covers {table_hz[0]:g}-{table_hz[-1]:g} Hz; "
            f"requested {frequencies_hz.min():g}-{frequencies_hz.max():g} Hz"
        )
    frequencies_hz = np.clip(frequencies_hz, table_hz[0], table_hz[-1])
    log_asd = np.interp(np.log(frequencies_hz), np.log(table_hz), np.log(table_asd))
    psd_per_hz = np.exp(2.0 * log_asd)
    assert psd_per_hz.shape == frequencies_hz.shape
    return psd_per_hz
