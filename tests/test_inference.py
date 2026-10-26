"""
Unit tests for the Keras and TensorFlow Lite inference engines.
"""

from pathlib import Path
import numpy as np
import pytest

from confidence_denoising.inference.engine import DenoisingEngine
from confidence_denoising.inference.tflite_engine import TFLiteDenoisingEngine


def test_keras_engine_single_window_and_full_signal():
    model_path = Path("models/confidence_model_1774696598.h5")
    if not model_path.exists():
        pytest.skip("Pretrained model not found at models/confidence_model_1774696598.h5")

    engine = DenoisingEngine(model_path)

    # 1. Single window prediction
    w = np.random.randn(256).astype(np.float32)
    denoised_win, conf = engine.predict_window(w)
    assert denoised_win.shape == (256,)
    assert 0.0 <= conf <= 1.0

    # 2. Full signal denoising
    sig = np.sin(np.linspace(0, 10, 600)).astype(np.float32)
    denoised_sig, confs = engine.denoise_signal(sig, mode="overlap_add")
    assert len(denoised_sig) == 600
    assert len(confs) == 600 - 256 + 1
    assert not np.isnan(denoised_sig).any()


def test_tflite_engine_prediction():
    tflite_path = Path("models/confidence_model_1774696598.tflite")
    if not tflite_path.exists():
        pytest.skip("TFLite model not found at models/confidence_model_1774696598.tflite")

    engine = TFLiteDenoisingEngine(tflite_path)
    assert engine.window_size == 256

    w = np.random.randn(256).astype(np.float32)
    denoised_win, conf = engine.predict_window(w)
    assert denoised_win.shape == (256,)
    assert 0.0 <= conf <= 1.0
    assert not np.isnan(denoised_win).any()
