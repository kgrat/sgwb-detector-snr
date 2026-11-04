"""Physical validation of the ORF module (checks against known results)."""

import numpy as np
import pytest

from sgwb_snr.orf import get_builtin_detector, tensor_orf

FREQS_HZ = np.array([0.1, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0])


def test_hanford_livingston_low_frequency_limit():
    # Known result: gamma_HL(f -> 0) = -0.89 (Christensen 1992;
    # Flanagan 1993; Allen & Romano 1999). Negative because the two
    # detectors' arms are rotated ~90 degrees relative to each other.
    gamma = tensor_orf(get_builtin_detector("H1"), get_builtin_detector("L1"), FREQS_HZ)
    assert gamma[0] == pytest.approx(-0.89, abs=0.005)


def test_colocated_coaligned_is_one():
    # Normalization convention: identical detectors give gamma = 1 at all f.
    gamma = tensor_orf(get_builtin_detector("H1"), get_builtin_detector("H1"), FREQS_HZ)
    np.testing.assert_allclose(gamma, 1.0, atol=1e-6)


def test_baseline_symmetry():
    # For the tensor ORF of an unpolarized isotropic background,
    # gamma_IJ(f) = gamma_JI(f).
    h1, l1 = get_builtin_detector("H1"), get_builtin_detector("L1")
    np.testing.assert_allclose(tensor_orf(h1, l1, FREQS_HZ), tensor_orf(l1, h1, FREQS_HZ), atol=1e-10)


def test_next_generation_builtins_are_rejected():
    for name in ["CE", "ET"]:
        with pytest.raises(ValueError):
            get_builtin_detector(name)


def test_bad_frequencies_rejected():
    h1, l1 = get_builtin_detector("H1"), get_builtin_detector("L1")
    with pytest.raises(ValueError):
        tensor_orf(h1, l1, np.array([0.0, 10.0]))
    with pytest.raises(ValueError):
        tensor_orf(h1, l1, np.array([10.0, 5.0]))
