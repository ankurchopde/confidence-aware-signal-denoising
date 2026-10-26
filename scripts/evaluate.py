#!/usr/bin/env python3
"""
Benchmark and Evaluation Script.
Evaluates the Confidence-Aware AI model against classical adaptive filters (LMS, NLMS)
across diverse signal waveforms and real-world noise environments.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from confidence_denoising.data.synthetic import (
    generate_linear_sine_mix,
    generate_square_wave,
    generate_sawtooth_wave,
    generate_am_modulated,
    generate_cubic_nonlinear,
    generate_am_nonlinear,
    generate_chaotic_nonlinear,
    add_gaussian_noise,
    add_impulse_noise,
    add_sine_interference,
    add_brown_noise,
    add_composite_nonlinear_noise,
)
from confidence_denoising.filtering.lms import lms_filter
from confidence_denoising.filtering.nlms import nlms_filter
from confidence_denoising.evaluation.metrics import (
    compute_snr_db,
    compute_mse,
    evaluate_performance,
)
from confidence_denoising.inference.engine import DenoisingEngine
from confidence_denoising.visualization.plotting import (
    plot_4panel_comparison,
    plot_metrics_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Confidence-Aware Denoising Model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/confidence_model_1774696598.h5",
        help="Path to trained model (.h5 or .keras)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save benchmark metrics JSON",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="figures/experiments",
        help="Directory to save benchmark figures",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="Save generated 4-panel evaluation plots",
    )
    return parser.parse_args()


def benchmark_suite(engine: DenoisingEngine, results_dir: Path, figures_dir: Path, save_plots: bool = True):
    fs = 1000
    duration = 1.0

    print("\n" + "=" * 65)
    print(" SUITE 1: EVALUATION ACROSS SIGNAL TOPOLOGIES (AWGN Noise)")
    print("=" * 65)

    signals_dict = {
        "Linear Sine Mix": generate_linear_sine_mix(duration, fs)[1],
        "Square Wave": generate_square_wave(duration, fs)[1],
        "Sawtooth Wave": generate_sawtooth_wave(duration, fs)[1],
        "AM Modulated": generate_am_modulated(duration, fs)[1],
        "Nonlinear (Sine+Cubic)": generate_cubic_nonlinear(duration, fs)[1],
    }

    suite1_results = {}
    for name, clean in signals_dict.items():
        noisy = add_gaussian_noise(clean, sigma=0.4, seed=42)
        y_lms = lms_filter(noisy, clean, mu=0.001)
        y_nlms = nlms_filter(noisy, clean, mu=0.1)
        y_model, confs = engine.denoise_signal(noisy, mode="overlap_add")

        snr_lms = compute_snr_db(clean, y_lms)
        snr_nlms = compute_snr_db(clean, y_nlms)
        snr_model = compute_snr_db(clean, y_model)

        mse_lms = compute_mse(clean, y_lms)
        mse_nlms = compute_mse(clean, y_nlms)
        mse_model = compute_mse(clean, y_model)

        suite1_results[name] = {
            "lms_snr_db": round(snr_lms, 2),
            "nlms_snr_db": round(snr_nlms, 2),
            "model_snr_db": round(snr_model, 2),
            "lms_mse": round(mse_lms, 4),
            "nlms_mse": round(mse_nlms, 4),
            "model_mse": round(mse_model, 4),
            "mean_confidence": round(float(np.mean(confs)), 4),
        }

        print(f"\n🔹 {name}:")
        print(f"   LMS   : MSE={mse_lms:.4f}, SNR={snr_lms:.2f} dB")
        print(f"   NLMS  : MSE={mse_nlms:.4f}, SNR={snr_nlms:.2f} dB")
        print(f"   MODEL : MSE={mse_model:.4f}, SNR={snr_model:.2f} dB (Conf={np.mean(confs):.3f})")

        if save_plots:
            fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
            plot_4panel_comparison(
                clean=clean,
                noisy=noisy,
                y_lms=y_lms,
                y_nlms=y_nlms,
                y_model=y_model,
                title=name,
                snr_lms=snr_lms,
                snr_nlms=snr_nlms,
                snr_model=snr_model,
                save_path=figures_dir / f"eval_signal_{fname}.png",
            )
            plt.close("all")

    print("\n" + "=" * 65)
    print(" SUITE 2: EVALUATION ACROSS NOISE TYPES (Linear Sine Mix)")
    print("=" * 65)

    _, clean_sine = generate_linear_sine_mix(duration, fs)
    noises_dict = {
        "Gaussian AWGN": add_gaussian_noise(clean_sine, sigma=0.4, seed=42),
        "Impulse Spikes": add_impulse_noise(clean_sine, ratio=0.02, seed=42),
        "50Hz Harmonic Hum": add_sine_interference(clean_sine, freq=50.0, amplitude=0.3, fs=fs),
        "Brown Noise Drift": add_brown_noise(clean_sine, amplitude=0.3, seed=42),
    }

    suite2_results = {}
    for noise_name, noisy in noises_dict.items():
        y_lms = lms_filter(noisy, clean_sine, mu=0.001)
        y_nlms = nlms_filter(noisy, clean_sine, mu=0.1)
        y_model, confs = engine.denoise_signal(noisy, mode="overlap_add")

        snr_lms = compute_snr_db(clean_sine, y_lms)
        snr_nlms = compute_snr_db(clean_sine, y_nlms)
        snr_model = compute_snr_db(clean_sine, y_model)

        mse_lms = compute_mse(clean_sine, y_lms)
        mse_nlms = compute_mse(clean_sine, y_nlms)
        mse_model = compute_mse(clean_sine, y_model)

        suite2_results[noise_name] = {
            "lms_snr_db": round(snr_lms, 2),
            "nlms_snr_db": round(snr_nlms, 2),
            "model_snr_db": round(snr_model, 2),
            "lms_mse": round(mse_lms, 4),
            "nlms_mse": round(mse_nlms, 4),
            "model_mse": round(mse_model, 4),
            "mean_confidence": round(float(np.mean(confs)), 4),
        }

        print(f"\n🔹 {noise_name}:")
        print(f"   LMS   : MSE={mse_lms:.4f}, SNR={snr_lms:.2f} dB")
        print(f"   NLMS  : MSE={mse_nlms:.4f}, SNR={snr_nlms:.2f} dB")
        print(f"   MODEL : MSE={mse_model:.4f}, SNR={snr_model:.2f} dB")

    print("\n" + "=" * 65)
    print(" SUITE 3: NONLINEAR SIGNALS UNDER COMPOSITE MIXED NOISE")
    print("=" * 65)

    nonlinear_dict = {
        "Cubic Nonlinearity": generate_cubic_nonlinear(duration, fs)[1],
        "AM-Modulated Nonlinear": generate_am_nonlinear(duration, fs)[1],
        "Chaotic-like Nonlinear": generate_chaotic_nonlinear(duration, fs)[1],
    }

    suite3_results = {}
    for nl_name, clean_nl in nonlinear_dict.items():
        noisy = add_composite_nonlinear_noise(clean_nl, fs=fs, seed=42)
        y_lms = lms_filter(noisy, clean_nl, mu=0.001)
        y_nlms = nlms_filter(noisy, clean_nl, mu=0.1)
        y_model, confs = engine.denoise_signal(noisy, mode="overlap_add")

        snr_lms = compute_snr_db(clean_nl, y_lms)
        snr_nlms = compute_snr_db(clean_nl, y_nlms)
        snr_model = compute_snr_db(clean_nl, y_model)

        mse_lms = compute_mse(clean_nl, y_lms)
        mse_nlms = compute_mse(clean_nl, y_nlms)
        mse_model = compute_mse(clean_nl, y_model)

        suite3_results[nl_name] = {
            "lms_snr_db": round(snr_lms, 2),
            "nlms_snr_db": round(snr_nlms, 2),
            "model_snr_db": round(snr_model, 2),
            "lms_mse": round(mse_lms, 4),
            "nlms_mse": round(mse_nlms, 4),
            "model_mse": round(mse_model, 4),
            "mean_confidence": round(float(np.mean(confs)), 4),
        }

        print(f"\n🔹 {nl_name}:")
        print(f"   LMS   : MSE={mse_lms:.4f}, SNR={snr_lms:.2f} dB")
        print(f"   NLMS  : MSE={mse_nlms:.4f}, SNR={snr_nlms:.2f} dB")
        print(f"   MODEL : MSE={mse_model:.4f}, SNR={snr_model:.2f} dB")

    # Generate summary chart
    plot_metrics_summary(
        results_by_signal=suite1_results,
        metric_name="snr_db",
        save_path=figures_dir / "benchmark_snr_comparison.png",
    )
    plt.close("all")

    full_results = {
        "signal_topologies": suite1_results,
        "noise_environments": suite2_results,
        "composite_nonlinear_suite": suite3_results,
    }

    json_path = results_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nAll benchmark results exported to: {json_path}")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {args.model_path}")
    engine = DenoisingEngine(args.model_path)
    benchmark_suite(engine, results_dir, figures_dir, save_plots=args.save_plots)


if __name__ == "__main__":
    main()
