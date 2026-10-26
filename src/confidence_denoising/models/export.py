"""
TensorFlow Lite export and quantization utilities for edge deployment.
"""

from pathlib import Path
from typing import Optional, Union
import tensorflow as tf
from tensorflow.keras import Model


def export_to_tflite(
    model: Model,
    output_path: Union[str, Path],
    quantize_dynamic: bool = False,
) -> Path:
    """
    Convert a trained Keras dual-head model to TensorFlow Lite (.tflite).

    Args:
        model: Trained Keras Model.
        output_path: Destination path for .tflite artifact.
        quantize_dynamic: If True, apply dynamic-range post-training quantization.

    Returns:
        Path to generated .tflite file.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize_dynamic:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_bytes = converter.convert()
    with open(out_p, "wb") as f:
        f.write(tflite_bytes)

    return out_p
