"""
Performance evaluation metrics: Signal-to-Noise Ratio (SNR), Mean Squared Error (MSE),
Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
"""

from typing import Dict, Union
import numpy as np


def compute_snr_db(
    clean: np.ndarray,
    processed: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR) in decibels (dB).

    Scales the processed signal amplitude to match clean signal peak amplitude
    before error calculation, consistent with experimental evaluation methodology:
        processed_scaled = processed * (max(|clean|) / (max(|processed|) + eps))
        noise = clean - processed_scaled
        SNR_dB = 10 * log10( sum(clean^2) / (sum(noise^2) + eps) )

    Args:
        clean: Ground truth clean signal.
        processed: Filtered or denoised signal.
        eps: Small epsilon for numerical stability.

    Returns:
        SNR in decibels (dB).
    """
    clean_arr = np.asarray(clean, dtype=np.float64).ravel()
    proc_arr = np.asarray(processed, dtype=np.float64).ravel()
    min_len = min(len(clean_arr), len(proc_arr))

    c = clean_arr[:min_len]
    p = proc_arr[:min_len]

    max_c = np.max(np.abs(c))
    max_p = np.max(np.abs(p)) + eps
    p_scaled = p * (max_c / max_p)

    noise = c - p_scaled
    p_sig = np.mean(c ** 2)
    p_noise = np.mean(noise ** 2) + eps

    return float(10.0 * np.log10(p_sig / p_noise))


def compute_mse(clean: np.ndarray, processed: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE) between clean and processed signals."""
    c = np.asarray(clean, dtype=np.float64).ravel()
    p = np.asarray(processed, dtype=np.float64).ravel()
    min_len = min(len(c), len(p))
    return float(np.mean((c[:min_len] - p[:min_len]) ** 2))


def compute_rmse(clean: np.ndarray, processed: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(compute_mse(clean, processed)))


def compute_mae(clean: np.ndarray, processed: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE)."""
    c = np.asarray(clean, dtype=np.float64).ravel()
    p = np.asarray(processed, dtype=np.float64).ravel()
    min_len = min(len(c), len(p))
    return float(np.mean(np.abs(c[:min_len] - p[:min_len])))


def evaluate_performance(
    clean: np.ndarray,
    processed: np.ndarray,
) -> Dict[str, float]:
    """
    Compute full metric suite (SNR, MSE, RMSE, MAE).

    Args:
        clean: Ground-truth clean signal.
        processed: Reconstructed signal.

    Returns:
        Dictionary of computed metrics.
    """
    return {
        "snr_db": compute_snr_db(clean, processed),
        "mse": compute_mse(clean, processed),
        "rmse": compute_rmse(clean, processed),
        "mae": compute_mae(clean, processed),
    }
