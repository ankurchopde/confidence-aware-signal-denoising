"""
Unit tests for evaluation metrics (SNR, MSE, RMSE, MAE).
"""

import numpy as np
import pytest

from confidence_denoising.evaluation.metrics import (
    compute_snr_db,
    compute_mse,
    compute_rmse,
    compute_mae,
    evaluate_performance,
)


def test_metrics_identical_signals():
    sig = np.array([1.0, -1.0, 0.5, -0.5, 0.2], dtype=np.float64)
    mse = compute_mse(sig, sig)
    rmse = compute_rmse(sig, sig)
    mae = compute_mae(sig, sig)
    snr = compute_snr_db(sig, sig)

    assert mse == 0.0
    assert rmse == 0.0
    assert mae == 0.0
    assert snr > 100.0  # Numerical infinite SNR capped by eps


def test_metrics_known_difference():
    clean = np.array([1.0, 2.0, 3.0, 4.0])
    # difference of 0.5 everywhere
    proc = clean + 0.5

    mse = compute_mse(clean, proc)
    rmse = compute_rmse(clean, proc)
    mae = compute_mae(clean, proc)

    assert np.isclose(mse, 0.25)
    assert np.isclose(rmse, 0.5)
    assert np.isclose(mae, 0.5)


def test_evaluate_performance_dictionary():
    clean = np.sin(np.linspace(0, 10, 100))
    proc = clean + 0.1 * np.cos(np.linspace(0, 10, 100))

    metrics = evaluate_performance(clean, proc)
    assert "snr_db" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert all(isinstance(v, float) for v in metrics.values())
