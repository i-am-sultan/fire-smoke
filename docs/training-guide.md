# 🏋️ Training Guide

## Overview

This guide explains how to train a YOLOv26 model using the Fire & Smoke Detection Training Pipeline. By the end of this guide, you will be able to configure your dataset, select appropriate starting weights, launch training, resume interrupted experiments, and interpret the generated outputs.

> [!IMPORTANT]
> **Prerequisites**
> Before proceeding, ensure that your dataset is fully prepared and validated according to the [Dataset Preparation Guide](dataset-preparation.md). You must have a valid `data.yaml` file, Python 3.9+, and the necessary CUDA dependencies installed.

---

## 💻 Hardware Recommendations

We recommend running training on a dedicated GPU for acceptable performance. Below are the hardware tiers based on dataset size.

*(Note: A "Large Dataset" is generally considered to be anything exceeding 50,000 images or containing highly complex, high-resolution frames).*

| Component | Minimum (Testing Only) | Recommended (Standard) | Large Dataset Training (>50k imgs) |
| --- | --- | --- | --- |
| **CPU** | 4+ Cores | 8+ Cores | 12+ Cores |
| **RAM** | 8 GB | 32 GB | 64 GB |
| **GPU** | CPU-only (Not recommended) | RTX 3060 or better | Multiple GPUs (e.g., RTX A6000) |
| **VRAM** | N/A | 8 GB+ | 16 GB+ per GPU |
| **Storage** | 10 GB Free | 50 GB SSD | 200+ GB NVMe SSD |

---

## 🛠️ Installation & Setup

1. **Clone the repository** using the internal organization URL:
```bash
git clone http://10.10.30.65:3000/trainee-ai-ml/fire-smoke-training-pipeline.git
cd fire-smoke-training-pipeline

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Verify installation:**
```bash
python scripts/training/train.py --help

```



---

## 📁 Dataset Configuration

Your dataset must follow the standard YOLO directory structure. Ensure your `data.yaml` file is correctly mapped to your directory paths:

```yaml
path: datasets/sample-dataset # Root directory of the dataset

train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: fire
  1: smoke

```

---

## 🚀 Launching Training

The pipeline supports transfer learning. Starting with a pre-trained Nano model (`yolov26n.pt`) is highly recommended for faster training and lightweight deployment.

### Quick Start

To launch a standard baseline training run, use the following command:

```bash
python scripts/training/train.py \
    --data datasets/sample-dataset/data.yaml \
    --results-dir results \
    --weights weights/yolov26n/yolov26n.pt \
    --name baseline_v1

```

### Key Training Parameters

* `--data`: Path to your dataset's YAML file.
* `--results-dir`: The root directory where all experiment logs, weights, and plots will be saved.
* `--name`: The name of your specific experiment (e.g., `baseline_v1`). Outputs will be saved under `results/FS-[name]`.
* `--epochs`: Number of training iterations. Default is `150`. (e.g., `--epochs 300`)
* `--batch-size`: Number of images processed at once. Adjust based on your GPU VRAM. Default is `192`.
* `--imgsz`: Target image resolution. Default is `640`. Larger sizes (e.g., `1280`) improve small-object detection but require exponentially more VRAM.
* `--patience`: Early stopping threshold. Training stops if no improvement is observed for this many epochs. Default is `20`.

### Device Selection

The pipeline defaults to `--device auto`, which automatically detects available CPU/GPU resources. You can override this manually:

* **Specific GPU:** `--device 0`
* **Multi-GPU:** `--device 0,1,2,3`
* **CPU Only:** `--device cpu` *(For testing/debugging only)*

---

## 🧬 Augmentations

The pipeline automatically applies domain-specific augmentations optimized for fire and smoke detection. *These are handled natively by the training script based on default hyperparameter configurations.*

* **HSV Augmentation:** Simulates lighting changes, camera differences, and weather variations.
* **Rotation (±10°):** Improves robustness to camera angle variations.
* **Mosaic:** Combines multiple images into a single training sample, improving context learning and small object detection.
* **MixUp:** Blends images together to act as regularization and reduce overfitting.
* **Close Mosaic:** Disables mosaic augmentation near the end of training so the final epochs resemble real, deployment-ready images.

---

## 🔄 Resuming & Debugging

**Debug Mode:** Use the `--debug` flag when validating a new dataset. This runs a rapid training loop on a tiny fraction of the data for minimal epochs to ensure the pipeline doesn't crash before you commit to a 20-hour run.

```bash
python scripts/training/train.py --data data.yaml --name debug_run --debug

```

**Resuming Training:** If your training is interrupted (e.g., power loss, server restart), you can seamlessly resume from the last saved epoch by pointing to `last.pt` and passing the `--resume` flag:

```bash
python scripts/training/train.py \
    --weights results/FS-baseline_v1/weights/last.pt \
    --resume

```

---

## 📊 Interpreting Outputs

Once training is complete, your results will be saved in `results/FS-[name]/`. Key files include:

* `weights/best.pt`: The model weights with the highest validation accuracy. **(Use this for deployment)**.
* `weights/last.pt`: The weights from the final completed epoch.
* `results.csv` & `results.png`: Epoch-by-epoch training logs and visual loss curves.
* `confusion_matrix.png`: Visualizes false positives and false negatives across your classes.

### Key Metrics to Watch

* **mAP@50 (Mean Average Precision):** Measures detection quality at an Intersection over Union (IoU) of 0.50. Higher is better.
* **mAP@50-95:** A much stricter metric requiring precise bounding boxes. Highly recommended for comparing different model versions.
* **Precision:** Measures **False Alarm Resistance**. High precision means fewer false fire alerts (e.g., confusing the sun for fire).
* **Recall:** Measures **Detection Coverage**. High recall means fewer missed fires. *For fire safety systems, Recall is typically prioritized over Precision.*

---

## 💡 Best Practices & Troubleshooting

### Best Practices

1. **Always Start With a Baseline:** Train using default parameters first. Record the results, and *then* modify hyperparameters.
2. **Change One Variable At A Time:** Do not change batch size, epochs, and image size simultaneously. You won't know which change caused the performance shift.
3. **Keep Experiment Names Meaningful:** Use descriptive names like `Nano-640-baseline` or `Nano-1280-NoMixup` rather than `test_1` or `final_final`.

### Common Issues

* **CUDA Out Of Memory:** Your GPU doesn't have enough VRAM for the current configuration. Reduce the `--batch-size` (e.g., from 64 to 32) or reduce the `--imgsz`.
* **Very Low mAP / 0 Detections:** Usually points to a dataset issue. Check that your `data.yaml` class mappings match your label files, and visualize your dataset to ensure bounding boxes are drawn correctly.
* **Overfitting:** If your Training mAP is going up but your Validation mAP is going down, the model is memorizing the data. Gather more diverse data, apply stricter augmentations, or run the semantic deduplication script to remove redundant frames.

---

*Next Step: Once you have a trained `best.pt` model, proceed to the [Evaluation Guide](evaluation-guide.md) to benchmark its accuracy and latency.*
