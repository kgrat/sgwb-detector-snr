"""Tensor overlap reduction functions (ORFs) for detector pairs.

Convention (Allen & Romano 1999; Thrane & Romano 2013): the tensor ORF is
normalized so that gamma(f) = 1 for co-located, co-aligned L-shaped
interferometers with 90-degree arms. This is the convention pygwb uses and
the one in arXiv:2502.12828 (their Eq. 1 with the 5/(8*pi) prefactor).

The geometry itself (sky integral over antenna patterns) is computed by
pygwb; this module fixes inputs, corrects for non-90-degree arms (see
tensor_orf) and enforces checks.
"""

import bilby
import numpy as np
from pygwb.baseline import Baseline

# bilby ships "CE" and "ET" detector files, but their sites do NOT match the
# configurations used in the CE/ET literature we are reproducing (bilby's CE
# file sits at the LIGO Hanford site, so CE-H1 gives gamma = 1 everywhere).
# Next-generation detectors must be defined explicitly from Gupta et al.
# (arXiv:2307.10421, Table II), never taken from bilby defaults.
_ALLOWED_BUILTIN = {"H1", "L1", "V1", "K1"}


def get_builtin_detector(name):
    """Return an empty bilby Interferometer for a current-generation detector.

    Raises for any detector whose built-in bilby geometry we have not
    validated for this project (including CE and ET).
    """
    if name not in _ALLOWED_BUILTIN:
        raise ValueError(
            f"Detector '{name}' is not allowed from bilby built-ins. "
            f"Allowed: {sorted(_ALLOWED_BUILTIN)}. Define next-generation "
            "detectors explicitly from Gupta et al. (arXiv:2307.10421) Table II."
        )
    return bilby.gw.detector.get_empty_interferometer(name)


def tensor_orf(detector_1, detector_2, frequencies_hz):
    """Tensor overlap reduction function gamma(f) for one baseline.

    Parameters
    ----------
    detector_1, detector_2 : bilby.gw.detector.Interferometer
    frequencies_hz : 1D array of strictly increasing, positive frequencies [Hz]

    Returns
    -------
    gamma : 1D array, same shape as frequencies_hz, dimensionless
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    assert frequencies_hz.ndim == 1, "frequencies_hz must be 1D"
    if np.any(frequencies_hz <= 0):
        raise ValueError("frequencies_hz must be strictly positive")
    if np.any(np.diff(frequencies_hz) <= 0):
        raise ValueError("frequencies_hz must be strictly increasing")

    baseline = Baseline(
        name=f"{detector_1.name}{detector_2.name}",
        interferometer_1=detector_1,
        interferometer_2=detector_2,
        frequencies=frequencies_hz,
    )
    gamma = np.asarray(baseline.tensor_overlap_reduction_function, dtype=float)

    # pygwb's tensor ORF (Nishizawa et al. 2009, arXiv:0903.0528, Eqs. 32-34)
    # uses only each detector's position, plane and arm bisector, i.e. it
    # treats every detector as L-shaped. With b the unit bisector and c the
    # in-plane unit vector along x - y, arms at opening angle beta give
    #   D = (x x - y y) / 2 = sin(beta) (b c + c b) / 2 = sin(beta) D_L,
    # where D_L is the L-shaped detector with the same plane and bisector.
    # The ORF is bilinear in D_1 and D_2, so the true ORF is
    # sin(beta_1) sin(beta_2) times pygwb's (unchanged for 90-degree arms).
    gamma = gamma * _arm_opening_sine(detector_1) * _arm_opening_sine(detector_2)

    assert gamma.shape == frequencies_hz.shape
    assert np.all(np.isfinite(gamma)), "non-finite ORF values"
    # With this normalization |gamma| <= sin(beta_1) sin(beta_2) <= 1.
    assert np.all(np.abs(gamma) <= 1.0 + 1e-9), "|gamma| > 1: normalization error"
    return gamma


def _arm_opening_sine(detector):
    """sin(beta), with beta the angle between the detector's two arms."""
    cos_beta = float(np.dot(detector.x, detector.y))
    if not -1.0 < cos_beta < 1.0:
        raise ValueError(f"Detector '{detector.name}' has parallel arms")
    return np.sqrt(1.0 - cos_beta**2)
