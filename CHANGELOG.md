# Changelog

All notable changes to the `confidence-aware-signal-denoising` project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-03-27

### Added
- **1D-CNN Multi-Task Architecture:** Implemented dual-head neural network for joint signal denoising regression (`reg_out`) and signal quality confidence classification (`cls_out`).
- **Confidence-Aware Online Adaptation:** Dynamic learning rate scheduling mechanism ($\eta = \eta_0 (0.5 + c)$) to track sensor waveform shifts without catastrophic forgetting during pure noise bursts.
- **Adaptive Filtering Baselines:** Full reference implementations for classical single-tap Least Mean Squares (LMS) and Normalized Least Mean Squares (NLMS) filters.
- **Synthetic Signal & Noise Generation Suite:**
  - Signal generators for Linear Sine Mixture, Square Wave, Sawtooth Wave, Amplitude Modulated (AM) waveform, Cubic Non-linear, AM Non-linear, and Chaotic Non-linear systems.
  - Realistic noise contamination engines for Additive White Gaussian Noise (AWGN), Impulse Spikes (Shot Noise), 50 Hz powerline harmonic hum, and Brownian low-frequency drift.
- **Evaluation Engine:** Metric calculation suite computing Signal-to-Noise Ratio (SNR in dB), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
- **Edge Deployment:** TensorFlow Lite export pipeline supporting both Float32 and dynamic-range post-training quantization for embedded microcontrollers and single-board computers.
- **Automated Test Suite:** 15 comprehensive unit tests covering data generation, windowing, reconstruction, metrics, model architectures, and inference engines.
- **CI/CD:** Automated GitHub Actions test workflow covering Python 3.9 through 3.12.
