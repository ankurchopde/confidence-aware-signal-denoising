"""
Unit tests for the Dual-Head 1D-CNN Model architecture.
"""

import numpy as np
import pytest
import tensorflow as tf

from confidence_denoising.models.cnn_dual_head import build_confidence_denoising_model


def test_model_build_and_shapes():
    window_size = 256
    model = build_confidence_denoising_model(window_size=window_size)

    # Check input shape
    assert model.input_shape == (None, window_size, 1)

    # Check dual output shapes
    # Output 0: Regression head (None, window_size)
    # Output 1: Classification head (None, 1)
    assert model.output_shape == [(None, window_size), (None, 1)]

    # Forward pass on small dummy batch
    dummy_input = np.random.randn(4, window_size, 1).astype(np.float32)
    reg_out, cls_out = model(dummy_input)

    assert reg_out.shape == (4, window_size)
    assert cls_out.shape == (4, 1)

    # Check classification sigmoid output range [0, 1]
    cls_vals = cls_out.numpy()
    assert np.all(cls_vals >= 0.0)
    assert np.all(cls_vals <= 1.0)


def test_model_parameter_count():
    model = build_confidence_denoising_model(window_size=256)
    total_params = model.count_params()
    # Verified ground truth parameter count is ~580,481
    assert 550_000 <= total_params <= 600_000
