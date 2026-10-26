# Model Artifacts and Checkpoint Registry

This directory contains trained neural network checkpoints and optimized edge deployment artifacts for the Confidence-Aware Sensor Signal Denoising framework.

---

## 1. Checkpoint Registry

| Artifact Filename | Format | Size | Description | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| `confidence_model_1774696598.h5` | HDF5 Keras | ~6.7 MB | Final fine-tuned dual-head model after online confidence adaptation | Verified |
| `confidence_model_1774696598.tflite` | TFLite (Float32) | ~2.3 MB | Optimized edge deployment model with dual output heads | Verified |
| `confidence_model_trained_1774696569.h5` | HDF5 Keras | ~6.7 MB | Checkpoint after 25 offline training epochs (prior to online adaptation) | Verified |

---

## 2. Model Tensor Specifications

- **Input Node:**
  - Shape: `(batch_size, 256, 1)`
  - Dtype: `float32`
  - Range: `[-1.0, 1.0]` (peak-normalized window)
- **Output 0 (`reg_out`):**
  - Shape: `(batch_size, 256)`
  - Dtype: `float32`
  - Activation: Linear
  - Representation: Denoised window waveform
- **Output 1 (`cls_out`):**
  - Shape: `(batch_size, 1)`
  - Dtype: `float32`
  - Activation: Sigmoid
  - Representation: Signal quality confidence score ($[0, 1]$)

---

## 3. Edge Deployment Benchmarks

- **Target Architecture:** ARM Cortex-A (Raspberry Pi 4 / 5), x86_64, Edge TPU
- **Inference Latency:** Sub-millisecond per window (~0.3 ms on standard CPU)
- **Storage Footprint:** 2.3 MB (Float32 TFLite) / ~0.6 MB (Dynamic Range Quantization)
