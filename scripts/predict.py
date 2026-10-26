#!/usr/bin/env python3
"""
Inference CLI script for sensor signal denoising and confidence evaluation.
Supports standard Keras models (.h5, .keras) and edge TensorFlow Lite models (.tflite).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from confidence_denoising.data.synthetic import generate_linear_sine_mix, add_gaussian_noise
from confidence_denoising.inference.engine import DenoisingEngine
from confidence_denoising.inference.tflite_engine import TFLiteDenoisingEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Run sensor denoising and confidence inference")
    parser.add_argument("--model-path", type=str, default="models/confidence_model_1774696598.h5", help="Path to model file")
    parser.add_argument("--tflite", action="store_true", help="Use TensorFlow Lite engine for inference")
    parser.add_argument("--input-file", type=str, default=None, help="Path to CSV or NPY input file containing 1D signal")
    parser.add_argument("--output-file", type=str, default="results/denoised_output.csv", help="Path to save denoised signal")
    parser.add_argument("--demo", action="store_true", default=False, help="Run on synthetic demo signal")
    parser.add_argument("--plot", action="store_true", help="Display or save visualization of output")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load or generate input signal
    if args.input_file:
        in_p = Path(args.input_file)
        if in_p.suffix == ".npy":
            noisy = np.load(str(in_p)).ravel()
        elif in_p.suffix == ".csv":
            df = pd.read_csv(in_p)
            num_cols = df.select_dtypes(include=[np.number]).columns
            noisy = df[num_cols[0]].values.ravel()
        else:
            raise ValueError(f"Unsupported input file format: {in_p.suffix}")
        clean = None
        print(f"Loaded {len(noisy)} samples from {args.input_file}")
    else:
        # Default: generate synthetic demo signal
        _, clean = generate_linear_sine_mix(duration=1.0, fs=1000)
        noisy = add_gaussian_noise(clean, sigma=0.4, seed=42)
        print("Generated synthetic test signal (1000 samples, AWGN sigma=0.4)")

    # 2. Run inference
    if args.tflite or str(args.model_path).endswith(".tflite"):
        print(f"Running edge inference with TFLite engine: {args.model_path}")
        engine = TFLiteDenoisingEngine(args.model_path)
        # Run window-by-window inference
        window_size = engine.window_size
        denoised_windows = []
        confs = []
        for start in range(0, len(noisy) - window_size + 1):
            w = noisy[start : start + window_size]
            dw, c = engine.predict_window(w)
            denoised_windows.append(dw)
            confs.append(c)

        from confidence_denoising.data.windowing import overlap_add_reconstruct
        denoised = overlap_add_reconstruct(np.array(denoised_windows), hop_size=1, window_size=window_size, target_length=len(noisy))
        denoised = denoised * (np.max(np.abs(noisy)) / (np.max(np.abs(denoised)) + 1e-12))
        avg_conf = float(np.mean(confs))
    else:
        print(f"Running inference with Keras engine: {args.model_path}")
        engine = DenoisingEngine(args.model_path)
        denoised, confs = engine.denoise_signal(noisy, mode="overlap_add")
        avg_conf = float(np.mean(confs))

    print(f"\nInference Complete!")
    print(f"Average Signal Confidence Score: {avg_conf:.4f} ({'Valid Signal' if avg_conf > 0.5 else 'Noise Dominated'})")

    # 3. Save output
    df_out = pd.DataFrame({"noisy": noisy, "denoised": denoised})
    if clean is not None:
        df_out["clean"] = clean
        from confidence_denoising.evaluation.metrics import compute_snr_db, compute_mse
        snr = compute_snr_db(clean, denoised)
        mse = compute_mse(clean, denoised)
        print(f"Reconstruction Quality: SNR = {snr:.2f} dB | MSE = {mse:.5f}")

    df_out.to_csv(output_path, index=False)
    print(f"Saved output dataframe to: {output_path}")

    # 4. Optional Plotting
    if args.plot:
        fig, ax = plt.subplots(figsize=(10, 4))
        if clean is not None:
            ax.plot(clean[:400], "k", lw=2, label="Clean Ground Truth")
        ax.plot(noisy[:400], "gray", alpha=0.5, label="Noisy Input")
        ax.plot(denoised[:400], "g", lw=2, label="Confidence-Aware AI Denoised")
        ax.set_title(f"Inference Prediction (Confidence: {avg_conf:.3f})")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = output_path.with_suffix(".png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved inference plot to: {plot_path}")


if __name__ == "__main__":
    main()
