# 🏗️ System Architecture

## Overview

The Fire & Smoke Detection Training Pipeline is an end-to-end framework designed for reproducibility and deployment readiness. It transforms raw, messy data into highly optimized inference engines with minimal friction.

The architecture is built on the philosophy that **data quality matters just as much as model size**, prioritizing rigorous dataset engineering before a single epoch is ever trained.

---

## 🗺️ High-Level Pipeline Flow

```text
┌─────────────────────────┐
│  Raw YOLO Datasets      │ (Multiple Sources)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│    standardize.py       │ Class Remapping | Label Normalization | YAML Gen
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│      cleaner.py         │ Corruption/Duplicate Removal | Background Balancing
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   deduplication.py      │ DINOv2 | FAISS | Semantic Deduplication
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│     eda_stats.py        │ Integrity Validation | Class Distribution Check
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│       train.py          │ YOLOv26 | Multi-GPU | Domain Augmentations
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│     benchmark.py        │ Accuracy (mAP/Recall) | Latency (CPU/GPU FPS)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│       export.py         │ ONNX | TensorRT (FP16/INT8)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│     inference.py        │ Video Testing | Quadrant (Split-Frame) Inference
└─────────────────────────┘

```

---

## ⚙️ Core Architecture Phases

### Phase 1: Dataset Engineering

Before training, all data passes through a multi-stage quality assurance workflow:

1. **Standardization (`standardize.py`):** Unifies class IDs (0: Fire, 1: Smoke) and normalizes coordinates.
2. **Cleaning (`cleaner.py`):** Purges corrupt images, zero-byte labels, and extreme aspect ratios.
3. **Semantic Deduplication (`deduplication.py`):** Uses **DINOv2** and **FAISS** to remove visually identical CCTV frames, vastly improving dataset diversity and reducing overfitting.
4. **Analysis (`eda_stats.py`):** Validates class balance and annotation integrity.

### Phase 2: Model Training

Powered by Ultralytics YOLOv26, the training module is built for stability and domain-specific performance.

* **Reproducibility:** Enforces deterministic training (Seed = 42).
* **Hardware Agnostic:** Automatically routes to CPU, Single-GPU, or Multi-GPU setups.
* **Domain Augmentations:** Leverages HSV shifts, Random Rotation, Mosaic, and MixUp specifically tuned to simulate lighting, weather, and camera variations common in fire scenarios.

### Phase 3: Evaluation & Benchmarking

A model is only useful if it meets real-world constraints. The pipeline evaluates two dimensions:

* **Accuracy:** mAP@50, mAP@50-95, Precision, and **Recall** (highly prioritized to avoid missed fire events).
* **Speed:** CPU Latency, GPU Latency, and overall FPS.

### Phase 4: Deployment & Inference

PyTorch (`.pt`) weights are strictly for research. For production, the pipeline exports optimized artifacts:

* **ONNX:** For cross-platform and CPU edge devices.
* **TensorRT:** For maximum throughput and low-latency NVIDIA GPU inference.

**Advanced Inference:** The pipeline supports **Quadrant Inference Mode** (`--split-frame`). It dynamically slices high-resolution video into a 2×2 grid, runs inference on each quadrant independently, and stitches the detections back together. *This drastically improves the detection of distant or tiny smoke plumes.*

---

## 📂 Repository Component Map

| Script | Core Responsibility |
| --- | --- |
| `standardize.py` | Normalizes dataset formats and generates valid YAMLs. |
| `cleaner.py` | Removes broken data and balances background ratios. |
| `deduplication.py` | Removes semantic (visual) duplicates via embeddings. |
| `eda_stats.py` | Generates dataset health statistics. |
| `visualize.py` | Draws bounding boxes to visually verify annotations. |
| `train.py` | Executes reproducible YOLOv26 training loops. |
| `benchmark.py` | Validates target metrics (mAP, Recall, FPS). |
| `export.py` | Converts PyTorch models to ONNX or TensorRT. |
| `inference.py` | Validates video deployment natively or in Quadrant mode. |

---

## 🔮 Future Roadmap

We are constantly improving the pipeline to support production-scale needs. Planned features include:

* Integration of additional YOLO variants.
* SAHI-based (Slicing Aided Hyper Inference) framework support.
* Live RTSP/Webcam stream processing.
* MLflow / Weights & Biases experiment tracking integration.
* Automated hyperparameter sweeps.