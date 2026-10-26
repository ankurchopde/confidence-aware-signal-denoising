"""
Unit tests for classical adaptive filters (LMS and NLMS).
"""

import numpy as np
import pytest

from confidence_denoising.filtering.lms import lms_filter
from confidence_denoising.filtering.nlms import nlms_filter


def test_lms_filter_shape_and_stability():
    clean = np.sin(np.linspace(0, 10, 500))
    noisy = clean + 0.2 * np.random.randn(500)

    y_lms = lms_filter(noisy, clean, mu=0.001)
    assert len(y_lms) == len(clean)
    assert not np.isnan(y_lms).any()
    assert not np.isinf(y_lms).any()


def test_nlms_filter_shape_and_stability():
    clean = np.sin(np.linspace(0, 10, 500))
    noisy = clean + 0.2 * np.random.randn(500)

    y_nlms = nlms_filter(noisy, clean, mu=0.05, eps=1e-6)
    assert len(y_nlms) == len(clean)
    assert not np.isnan(y_nlms).any()
    assert not np.isinf(y_nlms).any()


def test_zero_input_stability():
    zero_sig = np.zeros(100)
    y_lms = lms_filter(zero_sig, zero_sig)
    y_nlms = nlms_filter(zero_sig, zero_sig)
    assert np.allclose(y_lms, 0.0)
    assert np.allclose(y_nlms, 0.0)
