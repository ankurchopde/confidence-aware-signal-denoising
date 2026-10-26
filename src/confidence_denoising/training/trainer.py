"""
Offline training loop for the dual-head confidence-aware denoising model.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model


def train_offline(
    model: Model,
    x_train: np.ndarray,
    y_reg_train: np.ndarray,
    y_cls_train: np.ndarray,
    x_val: np.ndarray,
    y_reg_val: np.ndarray,
    y_cls_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 64,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    verbose: int = 2,
) -> keras.callbacks.History:
    """
    Train the dual-head model on the offline window dataset.

    Args:
        model: Compiled Keras Model.
        x_train: Training input windows (N_train, window_size, 1).
        y_reg_train: Clean target windows (N_train, window_size).
        y_cls_train: Binary classification labels (N_train, 1).
        x_val: Validation input windows.
        y_reg_val: Validation clean target windows.
        y_cls_val: Validation binary labels.
        epochs: Number of training epochs (default 25).
        batch_size: Mini-batch size (default 64).
        checkpoint_dir: Directory to save model checkpoint upon completion.
        verbose: Keras verbosity level.

    Returns:
        Keras History object.
    """
    callbacks = []
    if checkpoint_dir is not None:
        chk_p = Path(checkpoint_dir)
        chk_p.mkdir(parents=True, exist_ok=True)
        checkpoint_file = chk_p / "confidence_model_best.keras"
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_file),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            )
        )

    history = model.fit(
        x=x_train,
        y={"reg_out": y_reg_train, "cls_out": y_cls_train},
        validation_data=(x_val, {"reg_out": y_reg_val, "cls_out": y_cls_val}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    if checkpoint_dir is not None:
        final_file = chk_p / "confidence_model_final.keras"
        model.save(str(final_file))

    return history
