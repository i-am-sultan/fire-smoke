# 📈 Evaluation & Benchmarking Guide

## Overview

Training loss alone does not determine whether a model is suitable for the real world. The evaluation stage is critical for measuring detection accuracy, false alarm behavior, missed detections, inference latency, and throughput (FPS).

This repository provides a unified benchmarking pipeline to help you make informed deployment decisions.

**Standard Workflow:**
`Trained Model ──► Dataset Validation ──► Accuracy Metrics ──► Speed Benchmarking ──► Deployment Decision`

---

## 🚀 Running Benchmarks

The benchmarking script performs two independent evaluations: **Accuracy** (how well it detects) and **Speed** (how fast it runs).

### Standard Evaluation

To evaluate a trained model on a specific dataset, provide both the weights and the `data.yaml`:

```bash
python scripts/evaluation/benchmark.py \
    --trained-weights results/FS-baseline/weights/best.pt \
    --data datasets/sample-dataset/data.yaml

```

### Cross-Dataset Evaluation

A model may perform well on one dataset and poorly on another. Testing on multiple datasets helps measure true generalization. You can pass multiple YAML files to evaluate them sequentially:

```bash
python scripts/evaluation/benchmark.py \
    --trained-weights results/FS-baseline/weights/best.pt \
    --data datasets/site_a/data.yaml datasets/site_b/data.yaml

```

### Speed Benchmark Only

If you only want to measure hardware latency and FPS without running a full validation sweep, simply omit the `--data` argument:

```bash
python scripts/evaluation/benchmark.py \
    --trained-weights results/FS-baseline/weights/best.pt

```

---

## ⚙️ Advanced Benchmark Configurations

When running speed tests, the pipeline uses a synthetic input image. You can adjust the parameters to match your target deployment environment:

* **Image Size (`--imgsz`):** Default is `640`. Larger sizes (e.g., `1280`) typically increase accuracy and GPU memory usage, but decrease FPS.
* **Warmup Iterations (`--warmup`):** Default is `10`. Neural networks often run slower during initial iterations. Warmup prepares the hardware before the actual timing begins.
* **Benchmark Runs (`--runs`):** Default is `100`. Increasing this number (e.g., `--runs 500`) provides better statistical confidence for average latency.

*Example configuration:*

```bash
python scripts/evaluation/benchmark.py \
    --trained-weights results/FS-baseline/weights/best.pt \
    --imgsz 1280 \
    --warmup 20 \
    --runs 500

```

---

## 📊 Understanding Metrics

### Accuracy Metrics

* **mAP@50:** Mean Average Precision at an IoU (Intersection over Union) of 0.50. Measures how well predicted boxes overlap ground truth boxes. (>0.85 is generally excellent).
* **mAP@50-95:** A stricter metric evaluating IoU from 0.50 to 0.95. This is the **preferred metric** for comparing the overall quality of different models.
* **Precision:** `TP / (TP + FP)`. Measures how many of the model's detections were actually correct. High precision means **fewer false alarms**.
* **Recall:** `TP / (TP + FN)`. Measures how many real fires the model successfully detected. High recall means **fewer missed fires**.

> [!IMPORTANT]
> **The Golden Rule of Fire Detection:** > Missing a fire is exponentially worse than triggering a false alarm. Therefore, **Recall should generally be prioritized over Precision.** A model with 0.95 Recall and 0.92 Precision is almost always preferable to a model with 0.70 Recall and 0.98 Precision.

### Speed Metrics

* **Latency:** Time required to process one image (e.g., `5 ms/image`). Lower is better.
* **FPS (Frames Per Second):** `1 / Inference Time`. Higher is better.
* **CPU vs. GPU:** Latency is heavily dependent on hardware. A model might run at `13 FPS` on a CPU but achieve `250 FPS` on a dedicated GPU.

---

## ⚠️ Common Evaluation Mistakes

1. **Using Validation Data as Test Data:** The model's hyperparameters are often tuned based on the validation set. Always maintain a completely unseen, dedicated test set for your final benchmark.
2. **Comparing Across Different Datasets:** Comparing Model A (tested on Dataset A) against Model B (tested on Dataset B) is meaningless. Always benchmark competing models on the exact same dataset.
3. **Chasing Accuracy While Ignoring FPS:** A model with 99% accuracy is useless in a real-time safety system if it processes frames at 1 FPS.

---

## ✅ Pre-Deployment Checklist

Before promoting a model to the deployment stage, ensure you have:

* [ ] Evaluated the model on a strictly unseen test split.
* [ ] Recorded mAP@50 and mAP@50-95.
* [ ] Confirmed that **Recall** meets safety requirements.
* [ ] Confirmed that **Precision** is high enough to avoid alert fatigue.
* [ ] Recorded target hardware latency (CPU/GPU) and FPS.
* [ ] Saved all benchmark logs for historical tracking.

### Recommended Experiment Tracking

Tracking historical performance prevents regressions. Keep your results folder clean and documented:

```text
results/
├── FS-Nano-v1/
├── FS-Nano-v2/
└── benchmarks/
    ├── Nano-v1-report.md
    ├── Nano-v2-report.md
    └── version_comparison.xlsx

```

---

*Next Step: Once your model meets both accuracy and performance requirements, proceed to the [Deployment Guide](deployment-guide.md) to export ONNX/TensorRT engines and run production inference.*