"""
Signal normalization, sliding window framing, and signal reconstruction routines.
"""

from typing import List, Optional, Tuple, Union
import numpy as np


def standardize(signal: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Standardize a 1D signal to zero mean and unit variance.

    Args:
        signal: 1D signal array.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Standardized numpy float64 array.
    """
    arr = np.asarray(signal, dtype=np.float64)
    std = np.std(arr)
    return (arr - np.mean(arr)) / (std + eps)


def normalize_windows(
    windows: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Scale each window independently to unit peak amplitude ([-1, 1]).
    Preserves dynamic range within each window.

    Args:
        windows: 2D array of shape (num_windows, window_size).
        eps: Epsilon to prevent divide-by-zero.

    Returns:
        Normalized 2D array of same shape.
    """
    max_vals = np.max(np.abs(windows), axis=1, keepdims=True)
    return windows / (max_vals + eps)


def extract_windows(
    signal: np.ndarray,
    window_size: int = 256,
    hop_size: int = 1,
) -> np.ndarray:
    """
    Extract sliding windows from a 1D signal.

    Args:
        signal: 1D input array.
        window_size: Length of each window (default 256).
        hop_size: Step size between consecutive windows.

    Returns:
        2D array of shape (num_windows, window_size).
    """
    sig = np.asarray(signal, dtype=np.float32)
    n_samples = len(sig)
    if n_samples < window_size:
        raise ValueError(f"Signal length ({n_samples}) must be >= window_size ({window_size})")

    windows = []
    for start in range(0, n_samples - window_size + 1, hop_size):
        windows.append(sig[start : start + window_size])
    return np.array(windows, dtype=np.float32)


def overlap_add_reconstruct(
    window_predictions: np.ndarray,
    hop_size: int = 1,
    window_size: int = 256,
    target_length: Optional[int] = None,
) -> np.ndarray:
    """
    Reconstruct full-length signal from sliding window predictions via Overlap-Add (OLA).

    Args:
        window_predictions: 2D array of predictions (N, window_size).
        hop_size: Stride used during window extraction.
        window_size: Window length.
        target_length: Desired output length. If None, uses N + window_size - 1.

    Returns:
        1D reconstructed signal array.
    """
    num_windows = window_predictions.shape[0]
    total_len = (num_windows - 1) * hop_size + window_size
    recon = np.zeros(total_len, dtype=np.float64)
    count = np.zeros(total_len, dtype=np.float64)

    for i in range(num_windows):
        start = i * hop_size
        recon[start : start + window_size] += window_predictions[i].ravel()
        count[start : start + window_size] += 1.0

    count[count == 0] = 1.0
    result = recon / count

    if target_length is not None:
        result = result[:target_length]
    return result.astype(np.float32)


def center_sample_reconstruct(
    window_predictions: np.ndarray,
    window_size: int = 256,
    target_length: Optional[int] = None,
) -> np.ndarray:
    """
    Reconstruct full signal by taking the center sample of each sliding window (hop=1).
    Applies symmetric zero-padding at the boundaries.

    Args:
        window_predictions: 2D array of shape (N, window_size).
        window_size: Window length.
        target_length: Original signal length.

    Returns:
        1D reconstructed signal array.
    """
    center = window_size // 2
    y_center = window_predictions[:, center]
    if target_length is not None:
        pad_front = center
        pad_back = max(0, target_length - len(y_center) - pad_front)
        y_recon = np.pad(y_center, (pad_front, pad_back), mode="edge")[:target_length]
    else:
        y_recon = y_center
    return y_recon.astype(np.float32)


def build_window_dataset(
    sources: List[np.ndarray],
    window_size: int = 256,
    hop_size: int = 128,
    target_windows: int = 6000,
    signal_ratio: float = 0.65,
    noise_sigma: float = 0.3,
    noise_only_sigma: float = 0.8,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct multi-source window dataset with paired noisy/clean windows and classification labels.

    Label definition:
        1: Signal + Noise (active sensor information present)
        0: Pure Noise (inactive / spurious transmission)

    Args:
        sources: List of standardized 1D source signals.
        window_size: Window length (default 256).
        hop_size: Hop between successive windows (default 128).
        target_windows: Total desired window count (pads with noise if below).
        signal_ratio: Probability of selecting signal vs pure noise window.
        noise_sigma: Noise std added to signal windows.
        noise_only_sigma: Noise std for pure noise windows.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X_windows, Y_clean, Y_label).
    """
    rng = np.random.RandomState(seed)
    x_list, y_clean_list, y_label_list = [], [], []

    for s in sources:
        sig_len = len(s)
        for start in range(0, sig_len - window_size, hop_size):
            win_clean = s[start : start + window_size]
            if rng.rand() < signal_ratio:
                win_noisy = win_clean + noise_sigma * rng.randn(window_size)
                label = 1
            else:
                win_noisy = noise_only_sigma * rng.randn(window_size)
                label = 0
            x_list.append(win_noisy.astype(np.float32))
            y_clean_list.append(win_clean.astype(np.float32))
            y_label_list.append(label)

    x_arr = np.array(x_list, dtype=np.float32)
    y_clean_arr = np.array(y_clean_list, dtype=np.float32)
    y_label_arr = np.array(y_label_list, dtype=np.int32)

    # Pad with pure noise windows up to target_windows
    while x_arr.shape[0] < target_windows:
        w_noise = (noise_only_sigma * rng.randn(window_size)).astype(np.float32)
        x_arr = np.vstack([x_arr, w_noise[np.newaxis, :]])
        y_clean_arr = np.vstack([y_clean_arr, np.zeros((1, window_size), dtype=np.float32)])
        y_label_arr = np.append(y_label_arr, 0)

    return x_arr, y_clean_arr, y_label_arr
