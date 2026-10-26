"""
TensorFlow Lite inference engine for low-latency edge sensor denoising.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf


class TFLiteDenoisingEngine:
    """
    Inference runner utilizing the TensorFlow Lite interpreter.
    Suitable for embedded IoT gateways, microcontrollers, and single-board computers (Raspberry Pi).
    """

    def __init__(
        self,
        tflite_model_path: Union[str, Path],
    ) -> None:
        """
        Initialize the TFLite interpreter and allocate tensors.

        Args:
            tflite_model_path: Path to the .tflite model file.
        """
        self.model_path = Path(tflite_model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"TFLite model not found at {self.model_path}")

        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Determine regression (shape [1, 256]) and classification (shape [1, 1]) output indices
        self.reg_output_index = None
        self.cls_output_index = None

        for out in self.output_details:
            shape = out["shape"]
            if len(shape) == 2 and shape[1] == 1:
                self.cls_output_index = out["index"]
            elif len(shape) == 2 and shape[1] > 1:
                self.reg_output_index = out["index"]

        # Fallback if names/shapes differ
        if self.reg_output_index is None:
            self.reg_output_index = self.output_details[0]["index"]
        if self.cls_output_index is None and len(self.output_details) > 1:
            self.cls_output_index = self.output_details[1]["index"]

        input_shape = self.input_details[0]["shape"]
        self.window_size = int(input_shape[1])

    def predict_window(
        self,
        window: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Run edge inference on a single 1D sensor window.

        Args:
            window: 1D array of length `window_size`.

        Returns:
            Tuple of (denoised_window, confidence_probability).
        """
        w = np.asarray(window, dtype=np.float32).ravel()
        if len(w) != self.window_size:
            raise ValueError(f"Expected window of length {self.window_size}, got {len(w)}")

        peak = np.max(np.abs(w)) + 1e-12
        w_norm = (w / peak).reshape(1, self.window_size, 1).astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]["index"], w_norm)
        self.interpreter.invoke()

        reg_out = self.interpreter.get_tensor(self.reg_output_index).ravel()
        if self.cls_output_index is not None:
            cls_out = float(self.interpreter.get_tensor(self.cls_output_index).ravel()[0])
        else:
            cls_out = 1.0

        denoised_scaled = reg_out * peak
        return denoised_scaled, cls_out
