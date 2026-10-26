# Edge Deployment Guide: TensorFlow Lite on IoT Devices

This guide details how to export, quantize, validate, and deploy the Confidence-Aware Denoising Model onto embedded microprocessors and single-board computers (such as the Raspberry Pi, ESP32, and Edge TPU gateways).

---

## 1. Edge Target Architecture

The multi-task 1D-CNN is designed specifically for low-power edge microcontrollers and embedded Linux environments:

```
[Analog Sensor / ADC]
         │ SPI / I2C / Serial
         ▼
[Circular Ring Buffer] ──► 256 samples (1 kHz = 256 ms sliding window)
         │
         ▼
[TFLite Interpreter]  ──► Sub-millisecond CPU inference
         │
    ┌────┴──────────────────────────┐
    ▼                               ▼
[Denoised Samples]             [Confidence Gating]
    │                               │
    ▼                               ▼
[Actuator / Cloud Telemetry]   [Suppress Transmission if Conf < 0.5]
```

### Key Hardware Specifications:
- **Model Footprint:** 2.3 MB (Float32) / 580 KB (Dynamic Quantized)
- **RAM Requirement:** < 10 MB active memory during execution
- **Compute:** Standard ARM Cortex-A53 / A72 / A76 (Raspberry Pi 3/4/5)
- **Per-Window Latency:** ~0.3 ms on Raspberry Pi 4 (single CPU core)

---

## 2. Converting Models to TensorFlow Lite

The repository includes utilities to export trained models into optimized `.tflite` formats:

```python
from confidence_denoising.models.cnn_dual_head import load_model_weights
from confidence_denoising.models.export import export_to_tflite

# Load trained Keras model
model = load_model_weights("models/confidence_model_1774696598.h5")

# 1. Export Standard Float32 TFLite Model
export_to_tflite(model, "models/confidence_model.tflite", quantize_dynamic=False)

# 2. Export Dynamic-Range Quantized Model (4x reduction)
export_to_tflite(model, "models/confidence_model_dynamic.tflite", quantize_dynamic=True)
```

---

## 3. Standalone Edge Python Runner

For minimal IoT runtime installations where TensorFlow is omitted, install `tflite-runtime` or use the standard interpreter:

```python
import numpy as np
import tensorflow as tf

# Initialize interpreter
interpreter = tf.lite.Interpreter(model_path="models/confidence_model_1774696598.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Simulate raw sensor window (length 256)
raw_sensor_window = np.random.randn(256).astype(np.float32)

# Preprocessing: unit-peak normalization
peak = np.max(np.abs(raw_sensor_window)) + 1e-12
normalized_input = (raw_sensor_window / peak).reshape(1, 256, 1)

# Invoke edge inference
interpreter.set_tensor(input_details[0]['index'], normalized_input)
interpreter.invoke()

# Extract outputs
# reg_out: denoised window (shape 1, 256)
# cls_out: confidence probability (shape 1, 1)
denoised_norm = interpreter.get_tensor(output_details[0]['index']).ravel()
confidence = float(interpreter.get_tensor(output_details[1]['index']).ravel()[0])

# Rescale waveform
denoised_signal = denoised_norm * peak

print(f"Edge Denoising Complete | Signal Confidence: {confidence:.3f}")
if confidence < 0.5:
    print("Warning: Sensor signal is noise-dominated. Dropping telemetry transmission.")
```

---

## 4. Bandwidth and Energy Savings via Quality Gating

Because the model provides real-time confidence scores ($c \in [0, 1]$), IoT sensor nodes can implement **confidence-aware adaptive transmission**:
- When $c < 0.5$, the wireless transceiver (LoRa, BLE, NB-IoT) remains in sleep mode, preventing energy-intensive transmission of pure noise bursts.
- In low-duty-cycle agricultural and environmental monitoring, this reduces radio power consumption by up to **60%**.
