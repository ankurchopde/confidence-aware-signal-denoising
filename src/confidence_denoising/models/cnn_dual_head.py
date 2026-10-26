"""
Dual-Head 1D-CNN Architecture for Sensor Signal Denoising and Quality Classification.

Architecture:
    Input: (WINDOW, 1) raw sensor window
    Backbone:
        - Conv1D(filters=32, kernel_size=9, padding='same', activation='relu')
        - MaxPool1D(pool_size=2)
        - Conv1D(filters=64, kernel_size=7, padding='same', activation='relu')
        - MaxPool1D(pool_size=2)
        - Flatten
        - Dense(128, activation='relu')
    Output Heads:
        - Classification Head (cls_out):
            Dense(64, activation='relu') -> Dense(1, activation='sigmoid', name='cls_out')
            Estimates signal presence confidence / signal-to-noise state [0, 1].
        - Denoising Regression Head (reg_out):
            Dense(WINDOW, activation='linear', name='reg_out')
            Predicts the cleaned window waveform.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input


def build_confidence_denoising_model(
    window_size: int = 256,
    learning_rate: float = 1e-3,
    loss_weights: Optional[Dict[str, float]] = None,
) -> Model:
    """
    Construct and compile the dual-head 1D-CNN confidence-aware denoising model.

    Args:
        window_size: Length of input signal windows (default 256).
        learning_rate: Initial Adam optimizer learning rate.
        loss_weights: Dictionary containing weights for 'reg_out' and 'cls_out'.

    Returns:
        Compiled Keras Model instance.
    """
    if loss_weights is None:
        loss_weights = {"reg_out": 1.0, "cls_out": 0.2}

    inputs = Input(shape=(window_size, 1), name="signal_input")

    # Convolutional feature extraction backbone
    x = layers.Conv1D(32, kernel_size=9, padding="same", activation="relu", name="conv1d_1")(inputs)
    x = layers.MaxPool1D(2, name="maxpool_1")(x)
    x = layers.Conv1D(64, kernel_size=7, padding="same", activation="relu", name="conv1d_2")(x)
    x = layers.MaxPool1D(2, name="maxpool_2")(x)
    x = layers.Flatten(name="flatten")(x)
    shared_repr = layers.Dense(128, activation="relu", name="shared_dense")(x)

    # Classification / confidence head
    cls_branch = layers.Dense(64, activation="relu", name="cls_dense")(shared_repr)
    cls_out = layers.Dense(1, activation="sigmoid", name="cls_out")(cls_branch)

    # Regression / denoising head
    reg_out = layers.Dense(window_size, activation="linear", name="reg_out")(shared_repr)

    model = Model(inputs=inputs, outputs=[reg_out, cls_out], name="confidence_denoising_1dcnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"reg_out": "mse", "cls_out": "binary_crossentropy"},
        loss_weights=loss_weights,
        metrics={"cls_out": ["accuracy"]},
    )

    return model


def load_model_weights(
    model_or_path: Union[str, Path, Model],
    weights_path: Optional[Union[str, Path]] = None,
    compile_model: bool = False,
) -> Model:
    """
    Safely load a trained model or model weights from an HDF5 file.

    Args:
        model_or_path: Path to .h5 model file, or Model instance to load weights into.
        weights_path: Path to weights if model_or_path is a Model instance.
        compile_model: Whether to compile loaded model (default False).

    Returns:
        Keras Model with loaded weights.
    """
    if isinstance(model_or_path, (str, Path)):
        path_str = str(model_or_path)
        try:
            return keras.models.load_model(path_str, compile=compile_model)
        except Exception:
            # Fallback: instantiate architecture and load weights
            model = build_confidence_denoising_model()
            model.load_weights(path_str)
            return model
    else:
        if weights_path is not None:
            model_or_path.load_weights(str(weights_path))
        return model_or_path
