"""
Confidence-guided online fine-tuning and incremental adaptation.

Dynamically scales the optimizer learning rate based on the network's
internal confidence score (signal-presence probability from cls_out head):
    scaled_lr = base_lr * (0.5 + cls_confidence)

This prevents divergence or catastrophic forgetting during pure noise bursts
while enabling fast adaptation to legitimate sensor waveform shifts.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model


class OnlineAdapter:
    """
    Manages incremental learning and confidence-scaled parameter updates.
    """

    def __init__(
        self,
        model: Model,
        base_learning_rate: float = 1e-5,
    ) -> None:
        """
        Initialize the online adapter.

        Args:
            model: Pretrained Keras dual-head Model.
            base_learning_rate: Base learning rate for online updates (default 1e-5).
        """
        self.model = model
        self.base_lr = base_learning_rate

        # Ensure model is compiled with an optimizer
        try:
            _ = self.model.optimizer
        except Exception:
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.base_lr),
                loss={"reg_out": "mse", "cls_out": "binary_crossentropy"},
                loss_weights={"reg_out": 1.0, "cls_out": 0.2},
            )

    def _set_learning_rate(self, new_lr: float) -> bool:
        """Robustly update optimizer learning rate across Keras / TF versions."""
        opt = self.model.optimizer
        try:
            lr_obj = opt.learning_rate
            if hasattr(lr_obj, "assign"):
                lr_obj.assign(float(new_lr))
                return True
            if isinstance(lr_obj, tf.Variable):
                lr_obj.assign(float(new_lr))
                return True
        except Exception:
            pass

        try:
            opt.learning_rate = float(new_lr)
            return True
        except Exception:
            pass

        try:
            keras.backend.set_value(opt.learning_rate, float(new_lr))
            return True
        except Exception:
            pass

        return False

    @staticmethod
    def extract_confidence(predictions: Union[List[np.ndarray], Tuple[np.ndarray, ...]]) -> float:
        """Extract scalar confidence probability in [0, 1] from model output tuple."""
        try:
            if isinstance(predictions, (list, tuple)) and len(predictions) >= 2:
                cls_arr = np.asarray(predictions[1])
                return float(np.clip(cls_arr.ravel()[0], 0.0, 1.0))
            arr = np.asarray(predictions)
            return float(np.clip(arr.ravel()[-1], 0.0, 1.0))
        except Exception:
            return 0.5

    def update_window(
        self,
        x_win: np.ndarray,
        y_reg_win: np.ndarray,
        y_cls_win: np.ndarray,
    ) -> float:
        """
        Perform a single confidence-aware online update step on an incoming window.

        Args:
            x_win: Single window input of shape (1, window_size, 1).
            y_reg_win: Target reconstruction window (1, window_size).
            y_cls_win: Target class label (1, 1).

        Returns:
            Predicted confidence score used for scaling.
        """
        preds = self.model.predict(x_win, verbose=0)
        confidence = self.extract_confidence(preds)

        # Scale LR between 0.5x and 1.5x based on confidence
        scaled_lr = self.base_lr * (0.5 + confidence)
        self._set_learning_rate(scaled_lr)

        # Ensure correct regression target shape
        if y_reg_win.ndim == 1:
            y_reg_in = y_reg_win.reshape(1, -1)
        else:
            y_reg_in = y_reg_win

        if y_cls_win.ndim == 1:
            y_cls_in = y_cls_win.reshape(1, -1)
        else:
            y_cls_in = y_cls_win

        self.model.train_on_batch(
            x_win,
            {"reg_out": y_reg_in, "cls_out": y_cls_in},
        )

        # Reset optimizer to base learning rate
        self._set_learning_rate(self.base_lr)
        return confidence

    def adapt_stream(
        self,
        x_windows: np.ndarray,
        y_reg_windows: np.ndarray,
        y_cls_windows: np.ndarray,
        max_windows: int = 200,
        print_every: int = 50,
    ) -> List[float]:
        """
        Run incremental adaptation loop across a sequence of windows.

        Returns:
            List of confidence scores observed during adaptation.
        """
        n = min(len(x_windows), max_windows)
        confidences = []
        for i in range(n):
            x_w = x_windows[i : i + 1]
            yr_w = y_reg_windows[i : i + 1]
            yc_w = y_cls_windows[i : i + 1]
            conf = self.update_window(x_w, yr_w, yc_w)
            confidences.append(conf)
            if (i + 1) % print_every == 0 or (i + 1) == n:
                print(f"Online update {i + 1}/{n} | Confidence: {conf:.4f}")
        return confidences
