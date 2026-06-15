# 📊 Dataset Preparation Guide

## Overview

This guide explains how to prepare a custom dataset for training using the Fire & Smoke Detection Training Pipeline. By the end of this guide, you will be able to organize a YOLO dataset correctly, standardize class labels, clean low-quality samples, remove duplicate images, and validate dataset integrity before training.

---

## 📁 Supported Dataset Format

The pipeline expects datasets to be in the standard YOLO Detection format.

### Directory Structure

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml

```

### YOLO Label Format

Each image must have a corresponding `.txt` file with the exact same name (e.g., `image001.jpg` and `image001.txt`).

Inside the text file, annotations must follow this format:
`<class_id> <x_center> <y_center> <width> <height>`

*Example (`image001.txt`):*

```text
0 0.542 0.421 0.210 0.315
1 0.317 0.654 0.180 0.240

```

> [!NOTE]
> All bounding box coordinates must be normalized between `0.0` and `1.0`.

### Class Definitions

The training pipeline strictly relies on the following standard mapping:

* **0:** Fire
* **1:** Smoke

### Background Images (Negative Samples)

Background images (images with no fire or smoke) are intentionally supported and highly recommended to reduce false positives during deployment.

To include a background image, provide the image file alongside a **completely empty** `.txt` label file.

---

## 🛠️ Step-by-Step Preparation Workflow

### Step 1 — Download or Collect Data

Gather your raw data from CCTV footage, public datasets, or Roboflow exports. Before proceeding through the pipeline, perform a manual spot-check to ensure images are reasonably clear and labels exist.

### Step 2 — Standardize Dataset

Different datasets often use conflicting class IDs (e.g., Dataset A uses `0` for Fire, but Dataset B uses `1` for Fire).

Use the standardization script to unify all labels to our standard (`0: Fire`, `1: Smoke`):

```bash
python scripts/dataset/standardize.py \
    --dataset-dir datasets/my_dataset \
    --map 0:0 1:2 2:1 \
    --classes fire smoke other

```

*In the `--map` example above: Old Class `0` → New Class `0`, Old Class `1` → New Class `2`, Old Class `2` → New Class `1`.*

### Step 3 — Clean Dataset

> [!WARNING]
> **Destructive Action:** The cleaning script performs in-place modifications. Always create a backup of your dataset before running this step.

Run the cleaner to automatically remove corrupt images, zero-byte files, extreme aspect ratios, and exact duplicates:

```bash
python scripts/dataset/cleaner.py \
    --dataset-dir datasets/my_dataset

```

### Step 4 — Remove Semantic Duplicates

Video datasets frequently contain adjacent frames that are visually identical, which leads to model overfitting. Exact duplicate removal is not enough.

Run semantic deduplication, powered by DINOv2 embeddings and FAISS similarity search:

```bash
python scripts/dataset/deduplication.py \
    --dataset-dir datasets/my_dataset

```

*(The default similarity threshold is `0.985`. This significantly improves generalization and speeds up training).*

### Step 5 — Analyze Dataset

Generate statistics to ensure your dataset is balanced and healthy:

```bash
python scripts/dataset/eda_stats.py \
    --dataset-dir datasets/my_dataset

```

**Recommended Checks:** Ensure that **Missing Labels**, **Missing Images**, and **Malformed Annotations** all report `0`.

### Step 6 — Visualize Labels

Always inspect samples before dedicating hours to training. The visualization tool draws bounding boxes and class names on a random sample of images to help you spot incorrect boxes, wrong class IDs, or tiny labels.

```bash
python scripts/dataset/visualize.py \
    --image-dir datasets/my_dataset/train/images \
    --label-dir datasets/my_dataset/train/labels \
    --classes fire smoke

```

### Step 7 — Verify `data.yaml`

Ensure your `data.yaml` is correctly formatted and points to the right directories:

```yaml
path: datasets/my_dataset

train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: fire
  1: smoke

```

---

## ✅ Recommended Dataset Quality Checklist

Before moving to the training phase, verify the following:

* [ ] **Structure:** `train`, `valid`, and `test` directories all contain `images/` and `labels/` folders.
* [ ] **Annotations:** All labels follow standard YOLO formatting with normalized coordinates.
* [ ] **Class Mapping:** `0` is strictly Fire, `1` is strictly Smoke. No orphan labels exist.
* [ ] **Cleanliness:** Corrupt files, tiny images, and unreadable labels have been removed.
* [ ] **Diversity:** Semantic duplicates have been removed, and empty background images are included.
* [ ] **Validation:** A random sample of annotations has been visually inspected.

---

## 📈 Recommended Dataset Sizes

| Use Case | Minimum Images |
| --- | --- |
| **Prototype** | 1,000+ |
| **Small Project** | 5,000+ |
| **Production Model** | 20,000+ |
| **Large Scale Model** | 50,000+ |

> [!NOTE]
> Label quality, dataset diversity, and realistic deployment scenarios are vastly more important than sheer image volume.

---

## ⚠️ Common Mistakes

* **Missing Empty Labels:** Including a background `.jpg` without its corresponding empty `.txt` file will cause the pipeline to drop the image or throw an error.
* **Wrong Class Mapping:** Proceeding with `0 = Smoke` when the pipeline expects `0 = Fire`. Always standardize first.
* **Skipping Visualization:** Many training failures originate from malformed bounding boxes. Do not skip Step 6.

---

*Next Step: Once your dataset passes validation, continue to the [Training Guide](training-guide.md) to begin training your model.*