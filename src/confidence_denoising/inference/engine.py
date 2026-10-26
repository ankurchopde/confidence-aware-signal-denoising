"""
Keras-based inference engine for offline and streaming signal reconstruction.
"""

from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union
import numpy as np
from tensorflow.keras import Model

from confidence_denoising.data.windowing import (
    center_sample_reconstruct,
    overlap_add_reconstruct,
)
from confidence_denoising.models.cnn_dual_head import load_model_weights


class DenoisingEngine:
    """
    High-level inference engine for sensor signal denoising and confidence scoring.
    """

    def __init__(
        self,
        model_or_path: Union[str, Path, Model],
        window_size: int = 256,
    ) -> None:
        """
        Initialize the inference engine.

        Args:
            model_or_path: Trained Model instance or path to .h5/.keras file.
            window_size: Window length (default 256).
        """
        self.window_size = window_size
        if isinstance(model_or_path, (str, Path)):
            self.model = load_model_weights(model_or_path)
        else:
            self.model = model_or_path

    def predict_window(
        self,
        window: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Denoise a single raw signal window of length `window_size`.

        Args:
            window: 1D array of length `window_size`.

        Returns:
            Tuple of (denoised_window_1d, confidence_probability).
        """
        w = np.asarray(window, dtype=np.float32).ravel()
        if len(w) != self.window_size:
            raise ValueError(f"Expected window of length {self.window_size}, got {len(w)}")

        peak = np.max(np.abs(w)) + 1e-12
        w_norm = (w / peak).reshape(1, self.window_size, 1)

        preds = self.model.predict(w_norm, verbose=0)
        y_reg = np.asarray(preds[0]).ravel()
        y_cls = float(np.asarray(preds[1]).ravel()[0])

        denoised_scaled = y_reg * peak
        return denoised_scaled, y_cls

    def denoise_signal(
        self,
        noisy_signal: np.ndarray,
        mode: str = "overlap_add",
        hop_size: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Denoise an entire 1D sensor signal.

        Args:
            noisy_signal: 1D input array.
            mode: Reconstruction method: 'overlap_add' (recommended) or 'center_sample'.
            hop_size: Hop size for window extraction (default 1).

        Returns:
            Tuple of (denoised_signal, confidence_scores).
        """
        sig = np.asarray(noisy_signal, dtype=np.float32).ravel()
        n_samples = len(sig)
        if n_samples < self.window_size:
            raise ValueError(f"Signal length {n_samples} is shorter than window {self.window_size}")

        # Peak normalization for window framing
        peak = np.max(np.abs(sig)) + 1e-12
        sig_norm = sig / peak

        # Extract sliding windows
        windows = []
        for i in range(0, n_samples - self.window_size + 1, hop_size):
            windows.append(sig_norm[i : i + self.window_size].reshape(self.window_size, 1))
        x_in = np.array(windows, dtype=np.float32)

        y_pred, y_cls = self.model.predict(x_in, verbose=0, batch_size=128)
        confidences = np.asarray(y_cls).ravel()

        if mode == "overlap_add":
            recon = overlap_add_reconstruct(
                y_pred,
                hop_size=hop_size,
                window_size=self.window_size,
                target_length=n_samples,
            )
            # Rescale to match original amplitude scale
            recon = recon * peak / (np.max(np.abs(recon)) + 1e-12)
        elif mode == "center_sample":
            recon = center_sample_reconstruct(
                y_pred,
                window_size=self.window_size,
                target_length=n_samples,
            )
            recon = recon * peak
        else:
            raise ValueError(f"Unsupported reconstruction mode: {mode}")

        return recon.astype(np.float32), confidences

    def stream_generator(
        self,
        signal: np.ndarray,
        step: int = 50,
    ) -> Generator[Tuple[int, np.ndarray, np.ndarray, float], None, None]:
        """
        Yield sliding window inference steps simulating real-time streaming data.

        Yields:
            (start_idx, raw_window, denoised_window, confidence)
        """
        sig = np.asarray(signal, dtype=np.float32)
        for start in range(0, len(sig) - self.window_size + 1, step):
            raw_win = sig[start : start + self.window_size]
            denoised_win, conf = self.predict_window(raw_win)
            yield start, raw_win, denoised_win, conf
