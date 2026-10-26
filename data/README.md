# Dataset Architecture and Synthetic Signal Suite

This directory documents the data pipelines, sensor sources, synthetic signal suites, and noise generation methodologies used for training and evaluating the Confidence-Aware Denoising Model.

---

## 1. Data Pipeline Overview

The framework handles both real-world IoT sensor time-series and synthetic signal generators. Continuous signals are segmented into fixed-length windows:

- **Window Size ($L$):** 256 samples
- **Hop Size ($H$):** 128 samples (offline training, 50% overlap) / 1 sample (inference overlap-add)
- **Standardization:** Zero-mean, unit variance ($\mu=0, \sigma=1$) per continuous series
- **Window Peak Normalization:** Scaled by $\max(|x|) + \epsilon$ to bound amplitudes in $[-1, 1]$

---

## 2. Sensor Waveforms & Signal Topologies

The signal suite spans both linear harmonic oscillations and non-linear dynamic systems:

| Signal Type | Equation / Dynamics | Description |
| :--- | :--- | :--- |
| **Linear Sine Mix** | $0.7\sin(2\pi \cdot 5t) + 0.3\sin(2\pi \cdot 12t)$ | Multi-tone baseline harmonic signal |
| **Square Wave** | $\text{sgn}(\sin(2\pi \cdot 5t))$ | Abrupt transitions, broadband frequency harmonics |
| **Sawtooth Wave** | $2(ft - \lfloor ft + 1/2 \rfloor)$ | Asymmetric ramp waveform with periodic discontinuities |
| **AM Modulated** | $(1 + 0.5\sin(2\pi \cdot 2t))\sin(2\pi \cdot 10t)$ | Amplitude modulated carrier representing fading sensor telemetry |
| **Cubic Nonlinear** | $\sin(2\pi \cdot 6t) + 0.3\sin^3(2\pi \cdot 6t)$ | Harmonic distortion from sensor saturation / non-linear transducer |
| **AM Nonlinear** | $(1 + 0.5\sin(2\pi \cdot 2t))\sin^3(2\pi \cdot 10t)$ | Severe amplitude modulation coupled with non-linear distortion |
| **Chaotic Nonlinear** | $\tanh(\sin(2\pi \cdot 7t) + 0.4\sin(2\pi \cdot 13t))$ | Bounded non-linear saturation dynamic |
| **IoT Sensor Drift** | $0.5\sin(2\pi \cdot 0.2t) + 0.2\sin(2\pi \cdot 1.2t)$ | Slow thermal and environmental sensor drift |
| **Chirp Waveform** | $f_0=1\,\text{Hz} \to f_1=100\,\text{Hz}$ | Non-stationary linear frequency chirp sweep |
| **Pseudo-ECG** | $\text{sq}(2\pi \cdot 1.7t) \cdot \exp(-((t\bmod 1)-0.2)^2 / 0.01)$ | Biomedical pulse simulation |

---

## 3. Real-World Noise Models

To evaluate robustness against diverse physical interference, the following noise environments are modeled:

1. **Additive White Gaussian Noise (AWGN):** Thermal noise, $\sigma = 0.4$.
2. **Impulse Spikes (Shot Noise):** Sparse high-amplitude spikes (2% of samples, magnitude $0.5 - 1.5$) simulating communication glitches or ADC faults.
3. **Harmonic Interference:** 50 Hz powerline hum ($A=0.3$) at 1000 Hz sampling rate.
4. **Brownian (Brown) Noise:** Low-frequency random walk drift ($\int w(t) dt$).
5. **Composite Mixed Noise:** Simultaneous combination of Gaussian, 50 Hz hum, and impulse spikes.

---

## 4. Multi-Source Training Dataset Construction

During dataset generation:
1. Continuous signal streams are sliced into overlapping windows.
2. 65% of windows are paired with additive noise (Label `1`: Signal Present).
3. 35% are synthetic pure noise windows (Label `0`: Inactive / Spurious Transmission).
4. The dataset is expanded to $\ge 6,000$ windows and split 88% train / 12% validation.

External datasets such as the [Kaggle IoT Sensor Dataset](https://www.kaggle.com/datasets/caesarlupum/iot-sensordata) and [Adaptive Modulation Detection Dataset](https://www.kaggle.com/datasets/zyan1999/adaptive-modulation-detection-dataset) can be placed in this folder or specified via `--data-dir`.
