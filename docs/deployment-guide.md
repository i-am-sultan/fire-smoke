# 📦 Deployment Guide

## Overview

Training a model is only half the journey. This guide explains how to export your trained YOLOv26 fire and smoke detection models, convert them into high-performance deployment runtimes, execute video inference pipelines, and benchmark production performance.

By the end of this guide, you will be able to confidently deploy your model across multiple hardware ecosystems using **PyTorch**, **ONNX**, or **NVIDIA TensorRT**.

**Deployment Workflow:**
`Train Model ──► best.pt Checkpoint ──► Optimization/Export (ONNX/TensorRT) ──► Inference Validation ──► Production Deployment`

---

## 🏗️ Supported Deployment Runtimes

Choosing the correct deployment format depends heavily on your target hardware architecture and latency requirements:

### 1. PyTorch (`.pt`)

* **Best For:** Research, rapid prototyping, and model validation.
* **Advantages:** Native framework format; requires zero conversion or compilation steps.
* **Disadvantages:** High runtime overhead, heavy dependencies, and slower execution speeds.

### 2. ONNX (`.onnx`)

* **Best For:** Cross-platform deployment, CPU-based servers, and edge computing devices.
* **Advantages:** Framework-agnostic, portable, and supported by a wide variety of hardware accelerators (via ONNX Runtime).
* **Disadvantages:** Requires managing runtime versions; generally slower than hardware-specific compiled engines.

### 3. NVIDIA TensorRT (`.engine`)

* **Best For:** Production NVIDIA GPUs, real-time streams, and ultra-low latency requirements.
* **Advantages:** Maximum throughput (FPS), hardware-level optimizations, and significantly lower VRAM usage.
* **Disadvantages:** Strictly hardware-dependent (must be compiled on the target GPU architecture) and locked into the NVIDIA ecosystem.

---

## 🚀 Exporting Models for Production

The export pipeline converts your trained PyTorch checkpoint into optimized serialization formats. All operations are handled via the unified export script:

```bash
python scripts/deployment/export.py

```

### Standard Export Configurations

**Convert to ONNX Format**

```bash
python scripts/deployment/export.py \
    --trained-weights results/FS-baseline/weights/best.pt \
    --formats onnx

```

**Convert to TensorRT Format**

```bash
python scripts/deployment/export.py \
    --trained-weights results/FS-baseline/weights/best.pt \
    --formats tensorrt

```

**Simultaneous Multi-Format Export (Recommended)**

```bash
python scripts/deployment/export.py \
    --trained-weights results/FS-baseline/weights/best.pt \
    --formats onnx tensorrt

```

### Advanced Optimization Flags

* **FP16 Half-Precision Optimization (`--half`):** Highly recommended for TensorRT deployment on modern GPUs. It drops precision down to 16-bit floating points, slashing memory usage and drastically increasing FPS with negligible accuracy loss.
```bash
python scripts/deployment/export.py \
    --trained-weights results/FS-baseline/weights/best.pt \
    --formats tensorrt \
    --half

```


* **INT8 Quantization (`--int8`):** Compresses the model down to 8-bit integers for extreme throughput. *Note: INT8 can lead to slight drops in accuracy; always run a validation benchmark after choosing this option.*
* **Dynamic Input Shapes (`--dynamic`):** Enables the model to accept variable input dimensions rather than locking it to a static shape. *Note: This adds flexibility but may slightly reduce maximum inference optimization.*

### Choosing an Input Resolution

The baseline model defaults to an image size of `640`. Choosing the right input resolution forces a direct trade-off between speed and accuracy:

| Input Resolution | Primary Use Case | Trade-off Impact |
| --- | --- | --- |
| **640** | Standard real-time streams, edge hardware | Base performance, lowest latency |
| **960** | Medium-range monitoring, early smoke detection | Balanced speed and accuracy |
| **1280** | Long-range outdoor cameras, small smoke regions | High precision, requires capable GPU VRAM |
| **1920** | High-altitude or industrial facility surveillance | Maximum detection coverage, high latency |

---

## 🎥 Running Validation Inference

The repository includes an inference verification script to test your native or exported models on real media files.

```bash
python scripts/deployment/inference.py

```

### Core Inference Commands

**Run Native PyTorch Inference**

```bash
python scripts/deployment/inference.py \
    --weights results/FS-baseline/weights/best.pt \
    --source demo.mp4 \
    --results-dir deployment_results

```

**Run Optimized TensorRT Inference**

```bash
python scripts/deployment/inference.py \
    --weights models/best.engine \
    --source demo.mp4 \
    --results-dir deployment_results

```

### Primary Inference Flags

* **Confidence Threshold (`--conf`):** Controls the detection sensitivity. Default is `0.50`. Lowering this value (e.g., `--conf 0.30`) increases **Recall** (catches more fire/smoke but raises false alarms). Raising it increases **Precision** (fewer false alarms but risks missing a small fire).
* **Device Targeting (`--device`):** Explicitly run inference on CPU, a specific GPU, or allow automatic routing:
```bash
--device auto  # Automated routing
--device 0     # Direct target GPU ID
--device cpu   # Force fallback to CPU

```



---

## 🧩 Advanced Quadrant Inference Mode

Distant, small smoke plumes often slip under the detection threshold when an entire high-resolution frame is downscaled to `640x640`. To bypass this physical limitation, you can activate **Quadrant (Split-Frame) Inference Mode**:

```bash
python scripts/deployment/inference.py \
    --weights models/best.engine \
    --source demo.mp4 \
    --results-dir deployment_results \
    --split-frame

```

### Technical Workflow

1. **Split:** The pipeline logically divides incoming video frames into a `2×2` grid (4 separate quadrant sub-frames).
2. **Inference:** The model executes inference on each of the 4 quadrants independently.
3. **Merge:** Bounding boxes are dynamically stitched back together, applying Non-Maximum Suppression (NMS) across the boundaries.
4. **Output:** The combined high-resolution analytical video frame is saved.

> [!TIP]
> **Quadrant Trade-offs:** This mode significantly improves the detection of small, early-stage, or distant smoke plumes. However, because it runs inference four times per frame, throughput (FPS) will drop significantly, and memory consumption will increase.

---

## 📋 Standard Hardware Target Profiles

Match your deployment pipeline to one of our verified target configurations:

### ⚙️ Development & Prototyping

* **Format:** PyTorch (`.pt`)
* **Resolution:** 640
* **Device:** Local Workstation GPU

### ⚙️ Low-Power Edge Device

* **Format:** ONNX (`.onnx`)
* **Resolution:** 640
* **Device:** Edge Accelerator (e.g., Intel OpenVINO, CPU Gateway)

### ⚙️ Production NVIDIA Enterprise Server (High Throughput)

* **Format:** TensorRT (`.engine`)
* **Resolution:** 640
* **Precision Optimization:** FP16 (`--half`)
* **Device:** Data Center GPU (e.g., NVIDIA T4 / A10)

### ⚙️ Long-Range Surveillance (Maximum Safety & Coverage)

* **Format:** TensorRT (`.engine`)
* **Resolution:** 1280
* **Precision Optimization:** FP16 (`--half`)
* **Execution:** Quadrant Mode (`--split-frame`)

---

## ⚠️ Common Deployment Pitfalls

> [!WARNING]
> **Cross-Hardware TensorRT Failures:** TensorRT optimizations rely on hardware-specific compute capabilities. If you compile a `.engine` file on an RTX 3060 workstation and attempt to deploy it on an enterprise server Tesla T4 GPU, the runtime engine will crash. **Always build your TensorRT engines directly on the target deployment machine.**

* **Exporting Without Validation:** Runtimes occasionally optimize models differently. Never ship an exported engine to production without running a validation benchmark sweep to ensure precision scales properly.
* **Aggressive Confidence Thresholding:** Setting `--conf 0.90` inside a fire-safety system will suppress early, faint smoke wisps. Keep thresholds conservative to prioritize early notification.
* **Neglecting Low-Light / Edge-Case Media Testing:** Ensure your validation includes diverse deployment conditions (e.g., night streams, fog, and internal camera lens flares).

---

## ✅ Deployment Readiness Checklist

Before making your system live, ensure your chosen model variant satisfies the following parameters:

* [ ] Evaluated against a distinct, out-of-sample testing split.
* [ ] Recall performance complies with safety specifications.
* [ ] TensorRT or ONNX engine was successfully compiled directly on the production host.
* [ ] Exported inference accuracy was cross-checked with native framework outputs.
* [ ] Processing throughput meets stream framerate requirements (e.g., >30 FPS).
* [ ] Hand-validated via standard video or split-frame quadrant tests.
* [ ] Logging, alert hooks, and metrics tracking pipelines are active and verified.