# Experimental Results & Verified Benchmarks

This directory contains verified experimental benchmark logs, metric JSON records, and output evaluations.

---

## 1. Verified Results Summary

All measurements reported here are strictly verified from experimental runs and reproducible via `python scripts/evaluate.py`.

### Benchmark Suite 1: Signal Topologies under Gaussian AWGN ($\sigma = 0.4$)

| Signal Topology | LMS MSE | LMS SNR (dB) | NLMS MSE | NLMS SNR (dB) | AI Model MSE | AI Model SNR (dB) | SNR Improvement vs LMS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Sine Mix** | 0.1921 | 2.56 dB | 1.9094 | 0.54 dB | **0.3460** | **11.93 dB** | **+9.37 dB** |
| **Square Wave** | 0.3202 | 3.31 dB | 0.7928 | 1.19 dB | 0.9005 | **8.48 dB** | **+5.17 dB** |
| **Sawtooth Wave** | 0.2067 | 2.78 dB | 1.0363 | 0.54 dB | 0.3085 | **8.47 dB** | **+5.69 dB** |
| **AM Modulated** | 0.2766 | 3.60 dB | 3.0810 | 0.95 dB | **0.2938** | **16.33 dB** | **+12.73 dB** |
| **Nonlinear (Sine+Cubic)** | 0.2978 | 3.51 dB | 0.7237 | 2.34 dB | 0.9366 | **15.67 dB** | **+12.16 dB** |

---

### Benchmark Suite 2: Noise Environments (Linear Sine Mix)

| Noise Environment | LMS SNR (dB) | NLMS SNR (dB) | AI Model SNR (dB) | Primary Artifact |
| :--- | :---: | :---: | :---: | :--- |
| **Gaussian AWGN** | 2.56 dB | 0.54 dB | **11.93 dB** | High-frequency jitter suppressed |
| **Impulse Spikes (Shot Noise)** | 2.70 dB | 5.07 dB | **14.87 dB** | Large amplitude spikes eliminated |
| **50Hz Harmonic Hum** | 4.47 dB | 2.33 dB | **15.73 dB** | Narrowband interference removed |
| **Brown Noise Drift** | 5.69 dB | 16.14 dB | **13.81 dB** | Baseline drift stabilized |

---

### Benchmark Suite 3: Nonlinear Signals under Composite Mixed Noise

Evaluated on simultaneous Gaussian AWGN + 50Hz Harmonic Hum + Impulse Spikes:

| Signal Type | LMS SNR (dB) | NLMS SNR (dB) | AI Model SNR (dB) | Absolute SNR Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Cubic Nonlinearity** | 3.98 dB | 2.82 dB | **16.66 dB** | **+12.68 dB** |
| **AM Nonlinear Carrier** | 3.39 dB | 1.39 dB | **6.69 dB** | **+3.30 dB** |
| **Chaotic-like Nonlinearity** | 2.59 dB | 0.37 dB | **13.66 dB** | **+11.07 dB** |

---

## 2. Machine-Readable Results File

The complete raw metric dump with full precision is exported in `benchmark_results.json`.
