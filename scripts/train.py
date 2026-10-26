#!/usr/bin/env python3
"""
Training script for the Confidence-Aware AI Sensor Signal Denoising Model.
"""

import argparse
from pathlib import Path
import yaml
import numpy as np
import tensorflow as tf

from confidence_denoising.data.synthetic import generate_iot_chirp_ecg_suite
from confidence_denoising.data.windowing import (
    standardize,
    build_window_dataset,
)
from confidence_denoising.data.loader import (
    load_sensor_directory,
    prepare_train_test_split,
)
from confidence_denoising.models.cnn_dual_head import build_confidence_denoising_model
from confidence_denoising.models.export import export_to_tflite
from confidence_denoising.training.trainer import train_offline
from confidence_denoising.training.online_adaptation import OnlineAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Train Confidence-Aware Sensor Denoising Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing CSV sensor data")
    parser.add_argument("--epochs", type=int, default=25, help="Number of offline training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial Adam learning rate")
    parser.add_argument("--window-size", type=int, default=256, help="Window size in samples")
    parser.add_argument("--target-windows", type=int, default=6000, help="Target total training windows")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for saved models")
    parser.add_argument("--online-adapt", action="store_true", help="Run online confidence-scaled fine-tuning")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== [1/5] Preparing Training Sources ===")
    sources = []
    if args.data_dir and Path(args.data_dir).exists():
        sources.extend(load_sensor_directory(args.data_dir))
        print(f"Loaded {len(sources)} series from {args.data_dir}")

    # Add synthetic IoT suite to ensure robust diversity
    synth = generate_iot_chirp_ecg_suite(t_max=10.0, num_samples=50000)
    for name, s in synth.items():
        sources.append(standardize(s))
    print(f"Total source signal series: {len(sources)}")

    print("=== [2/5] Framing Windows and Generating Dataset ===")
    x_windows, y_clean, y_labels = build_window_dataset(
        sources=sources,
        window_size=args.window_size,
        hop_size=args.window_size // 2,
        target_windows=args.target_windows,
        seed=0,
    )
    print(f"Generated windows: {x_windows.shape} (Signal: {int(y_labels.sum())}, Noise: {int((y_labels==0).sum())})")

    x_train, x_val, yr_train, yr_val, yc_train, yc_val = prepare_train_test_split(
        x_windows, y_clean, y_labels, test_size=0.12, random_state=42
    )
    print(f"Train split: {x_train.shape} | Validation split: {x_val.shape}")

    print("=== [3/5] Building and Compiling Dual-Head 1D-CNN ===")
    model = build_confidence_denoising_model(
        window_size=args.window_size,
        learning_rate=args.lr,
        loss_weights={"reg_out": 1.0, "cls_out": 0.2},
    )
    model.summary()

    print("=== [4/5] Training Model Offline ===")
    history = train_offline(
        model=model,
        x_train=x_train,
        y_reg_train=yr_train,
        y_cls_train=yc_train,
        x_val=x_val,
        y_reg_val=yr_val,
        y_cls_val=yc_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=output_dir,
    )

    if args.online_adapt:
        print("=== Running Confidence-Aware Online Adaptation Pass ===")
        adapter = OnlineAdapter(model, base_learning_rate=1e-5)
        adapter.adapt_stream(x_train, yr_train, yc_train, max_windows=200, print_every=50)

    print("=== [5/5] Exporting Model Artifacts ===")
    model_save_path = output_dir / "confidence_model.keras"
    model.save(str(model_save_path))
    print(f"Saved native Keras model: {model_save_path}")

    # TFLite float conversion
    tflite_path = output_dir / "confidence_model.tflite"
    export_to_tflite(model, tflite_path, quantize_dynamic=False)
    print(f"Saved float32 TFLite model: {tflite_path} ({tflite_path.stat().st_size / 1024:.1f} KB)")

    # TFLite dynamic quantization
    tflite_dyn_path = output_dir / "confidence_model_dynamic.tflite"
    export_to_tflite(model, tflite_dyn_path, quantize_dynamic=True)
    print(f"Saved dynamic quant TFLite: {tflite_dyn_path} ({tflite_dyn_path.stat().st_size / 1024:.1f} KB)")

    print("\nTraining workflow completed successfully!")


if __name__ == "__main__":
    main()
