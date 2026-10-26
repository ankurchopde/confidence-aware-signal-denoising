# Reproducibility Guide

This guide provides end-to-end instructions for reproducing all experimental results, training models from scratch, running automated unit tests, and verifying numerical metrics.

---

## 1. System Requirements

- **Operating System:** Linux (Ubuntu 20.04+, Fedora 38+, Debian 11+), macOS 12+, or Windows 10/11 (WSL2 recommended)
- **Python Version:** Python 3.9, 3.10, 3.11, or 3.12
- **Hardware:** Standard x86_64 or ARM64 CPU (No GPU required for evaluation or testing; GPU accelerates training)
- **Disk Space:** ~1.5 GB for virtual environment and model artifacts

---

## 2. Environment Setup

### Option A: Standard Virtualenv

```bash
# Clone the repository
git clone https://github.com/ankurchopde/confidence-aware-signal-denoising.git
cd confidence-aware-signal-denoising

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install package with development dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 3. Running Automated Tests

Verify package integrity and algorithmic calculations:

```bash
pytest -v
```

All 15 automated test cases should pass in under 3 seconds.

---

## 4. Reproducing Experimental Benchmarks

To execute the multi-suite benchmark comparison against LMS and NLMS filters:

```bash
python scripts/evaluate.py --model-path models/confidence_model_1774696598.h5
```

This command will:
1. Generate test signals (Linear Sine Mix, Square, Sawtooth, AM Modulated, Cubic Non-linear).
2. Apply 4 noise regimes and composite mixed interference.
3. Compute MSE, SNR, and confidence scores across all methods.
4. Export raw results to `results/benchmark_results.json`.
5. Save comparison plots to `figures/experiments/`.

---

## 5. Training Models from Scratch

To train the multi-task 1D-CNN from scratch on the synthetic training suite:

```bash
python scripts/train.py --epochs 25 --batch-size 64 --target-windows 6000 --online-adapt
```

This procedure generates a paired window dataset, trains for 25 epochs, executes 200 steps of online confidence-scaled fine-tuning, and exports both `.keras` and `.tflite` artifacts.

---

## 6. Single-Signal Inference Example

Denoise a synthetic demo signal:

```bash
python scripts/predict.py --demo --plot
```

Run edge inference using the lightweight TensorFlow Lite runtime:

```bash
python scripts/predict.py --demo --tflite --model-path models/confidence_model_1774696598.tflite
```
