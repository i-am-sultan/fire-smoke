# 🔄 Pipeline Orchestrator Configuration Guide

This document provides comprehensive documentation for the `pipeline.py` orchestrator and `pipeline.yaml` configuration file.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration Schema](#configuration-schema)
4. [Stage Reference](#stage-reference)
5. [Usage Examples](#usage-examples)
6. [Advanced Topics](#advanced-topics)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### What is the Pipeline Orchestrator?

The `pipeline.py` script is a **unified command-line orchestrator** that chains together all dataset preparation, training, evaluation, and deployment stages defined in a YAML configuration file.

**Key Benefits:**
- ✅ **Single command**: Run entire workflow with `python pipeline.py -c pipeline.yaml`
- ✅ **Flexible composition**: Enable/disable stages as needed
- ✅ **Reproducible**: Version-control your entire experiment configuration
- ✅ **Automatic path handling**: Inter-stage dependencies managed automatically
- ✅ **Error recovery**: Pipeline stops immediately on failure (no silent cascading errors)

### Architecture

```
pipeline.yaml (Configuration)
        ↓
pipeline.py (Orchestrator)
        ↓
┌───────────────────────────┐
│   Stage Execution Loop    │
├───────────────────────────┤
│ 1. Standardize (optional) │
│ 2. Clean (optional)       │
│ 3. Deduplicate (optional) │
│ 4. EDA (optional)         │
│ 5. Train (required)       │
│ 6. Benchmark (optional)   │
│ 7. Export (optional)      │
│ 8. Inference (optional)   │
└───────────────────────────┘
        ↓
   Results & Logs
```

---

## Quick Start

### Minimal Setup

```bash
# 1. Use the default configuration
cp pipeline.yaml pipeline.yaml.bak  # Backup original

# 2. Edit pipeline.yaml to match your dataset location
#    Update: dataset.dir, training.data_yaml, training.weights

# 3. Run the pipeline
python pipeline.py -c pipeline.yaml

# Pipeline will execute stages in order and stop on first failure
```

### Running Specific Stages

Edit `pipeline.yaml` to include only the stages you want:

```yaml
pipeline:
  stages:
    - standardize    # Keep only these stages
    - clean
    - deduplicate
    # - eda          # Comment out to skip
    # - train        # Comment out to skip
    # - benchmark
    # - export
    # - inference
```

Then run: `python pipeline.py -c pipeline.yaml`

---

## Configuration Schema

### Complete `pipeline.yaml` Structure

```yaml
# ==============================================================================
# PIPELINE ORCHESTRATOR CONFIGURATION
# ==============================================================================
#
# This YAML file defines which stages to execute and all parameters for each.
# Uncomment/comment stages to control what runs.
# All paths are relative to the repository root directory.

pipeline:
  stages:
    # Stage execution order (uncomment to enable):
    - standardize      # 1. Standardize dataset & class labels
    - clean            # 2. Remove corrupted files & invalid annotations
    - deduplicate      # 3. Remove semantic duplicates (DINOv2)
    - eda              # 4. Generate EDA statistics & visualizations
    - train            # 5. Train YOLOv26 model
    - benchmark        # 6. Benchmark inference performance
    - export           # 7. Export to ONNX & TensorRT
    - inference        # 8. Run inference on video/images

# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================
dataset:
  # Path to dataset directory (must contain train/, valid/, test/ subdirs)
  dir: "datasets/sample-dataset/"

  # Class labels (must match your dataset)
  classes:
    - "fire"
    - "smoke"

  # Optional: Class remapping (e.g., map old class IDs to new ones)
  # Format: ["old_id:new_id", "old_id:new_id"]
  # Example: ["0:0", "1:1", "2:1"] means class 2 becomes class 1
  # map:
  #   - "0:0"
  #   - "1:1"
  #   - "2:1"

  # Deduplication settings (only used by 'deduplicate' stage)
  dedup_threshold: 0.985      # Similarity threshold (0-1, higher = stricter)
  dedup_batch_size: 64        # Batch size for FAISS processing

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================
training:
  # Path to dataset YAML (YOLOv26 format)
  data_yaml: "datasets/sample-dataset/data.yaml"

  # Pre-trained weights path (nano, small, medium, large, x-large available)
  weights: "weights/yolov26n/yolo26n.pt"

  # Output directory for training results
  results_dir: "results/"

  # Experiment name (used to create results/FS-{name}/ directory)
  name: "FS-Pipeline-Run"

  # Training hyperparameters
  epochs: 100                  # Number of training epochs
  patience: 20                 # Early stopping patience (epochs without improvement)
  imgsz: 640                   # Input image size (must be multiple of 32)
  batch_size: 64               # Batch size for training
  workers: 8                   # Number of data loading workers (0 for CPU)
  device: "auto"               # Device: "auto" / "cpu" / "0" / "0,1,2,3" (multi-GPU)
  debug: true                  # If true, runs only 2 epochs for testing

# ==============================================================================
# BENCHMARK CONFIGURATION
# ==============================================================================
# Settings for inference latency & FPS benchmarking
benchmark:
  warmup: 10                   # Number of warmup iterations (not counted)
  runs: 100                    # Number of benchmark runs to average

# ==============================================================================
# EXPORT CONFIGURATION
# ==============================================================================
# Settings for model export (PyTorch → ONNX → TensorRT)
export:
  # Export formats: can include "onnx", "tensorrt", "tflite", "pb"
  formats:
    - "onnx"
    - "tensorrt"

  # Quantization options
  half: true                   # Use FP16 quantization (smaller, faster on GPUs)
  int8: false                  # Use INT8 quantization (requires calibration data)
  dynamic: false               # Use dynamic input shapes (flexibility vs performance)

  # ONNX opset version (higher = newer features, but less compatibility)
  opset: 17

  # Device for export (usually same as training device)
  device: "0"

# ==============================================================================
# INFERENCE CONFIGURATION
# ==============================================================================
# Settings for running inference on video/images
inference:
  # Source: path to video file, image file, or image directory
  source: "videos/test.mp4"

  # Confidence threshold (detections below this are filtered)
  conf: 0.5

  # Split-frame mode: divides frames into 2×2 grid for better small object detection
  # Trade-off: 4× slower but better for distant/small smoke detection
  split_frame: false
```

### Configuration Parameter Reference

#### Dataset Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dir` | str | ✓ | — | Path to dataset root directory |
| `classes` | list | ✓ | — | List of class names (e.g., `["fire", "smoke"]`) |
| `map` | list | ✗ | — | Class remapping (format: `["0:0", "1:2"]`) |
| `dedup_threshold` | float | ✗ | 0.985 | Similarity threshold for deduplication (0.0-1.0) |
| `dedup_batch_size` | int | ✗ | 32 | Batch size for FAISS processing |

#### Training Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `data_yaml` | str | ✓ | — | Path to YOLOv26 data.yaml |
| `weights` | str | ✓ | — | Path to pre-trained weights |
| `results_dir` | str | ✓ | — | Directory to save training results |
| `name` | str | ✓ | — | Experiment name (used in output paths) |
| `epochs` | int | ✗ | 150 | Number of training epochs |
| `patience` | int | ✗ | 20 | Early stopping patience |
| `imgsz` | int | ✗ | 640 | Input image size (32, 64, ..., 1024) |
| `batch_size` | int | ✗ | 16 | Batch size (adjust down if OOM) |
| `workers` | int | ✗ | 8 | Data loading workers (0 for CPU) |
| `device` | str | ✗ | "auto" | Device: "auto", "cpu", "0", "0,1,2,3" |
| `debug` | bool | ✗ | false | If true, run only 2 epochs |

#### Benchmark Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `warmup` | int | ✗ | 10 | Warmup iterations (not counted in averages) |
| `runs` | int | ✗ | 100 | Number of benchmark runs to average |

#### Export Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `formats` | list | ✗ | ["onnx"] | Export formats: onnx, tensorrt, tflite, pb |
| `half` | bool | ✗ | true | Enable FP16 quantization |
| `int8` | bool | ✗ | false | Enable INT8 quantization |
| `dynamic` | bool | ✗ | false | Enable dynamic input shapes |
| `opset` | int | ✗ | 17 | ONNX opset version (11-20) |
| `device` | str | ✗ | "0" | Export device |

#### Inference Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source` | str | ✓ | — | Video/image path or directory |
| `conf` | float | ✗ | 0.5 | Confidence threshold (0.0-1.0) |
| `split_frame` | bool | ✗ | false | Enable quadrant split-frame mode |

---

## Stage Reference

### Stage: `standardize`

**Purpose:** Remap dataset classes and generate YOLOv26-compatible YAML files.

**Requires:**
- Dataset directory with `train/`, `valid/`, `test/` subdirectories
- Each subdirectory must have `images/` and `labels/` folders
- Class mapping (optional)

**Produces:**
- `data.yaml` (YOLO dataset config)
- Remapped label files

**Example:**
```bash
python scripts/dataset/standardize.py --dataset-dir ./datasets/sample-dataset/ \
  --classes fire smoke \
  --map "0:0" "1:1" "2:1"
```

---

### Stage: `clean`

**Purpose:** Remove corrupted images, invalid annotations, and other artifacts.

**Requires:**
- Standardized dataset (run `standardize` first)

**Produces:**
- Clean dataset in the same directory
- Log file: `cleaning_report.json`

**Checks Performed:**
- ✓ Image corruption detection
- ✓ Missing corresponding annotations
- ✓ Extreme aspect ratios (filtered based on config)
- ✓ Tiny objects (below pixel threshold)
- ✓ Invalid YOLO format annotations

**Example:**
```bash
python scripts/dataset/cleaner.py --dataset-dir ./datasets/sample-dataset/
```

---

### Stage: `deduplicate`

**Purpose:** Remove semantically similar images using DINOv2 embeddings and FAISS.

**Requires:**
- Cleaned dataset (run `clean` first)
- GPU with sufficient VRAM (~20GB for large datasets)

**Produces:**
- Deduplicated dataset
- `deduplication_report.json` (similarity scores)

**Algorithm:**
1. Extract DINOv2 features for all images
2. Build HNSW index (FAISS)
3. Find similar neighbors above threshold
4. Remove lower-confidence images from similar pairs

**Threshold Guide:**
- `0.95` = Very strict (removes many images)
- `0.985` = Balanced (recommended)
- `0.99` = Permissive (keeps most images)

**Example:**
```bash
python scripts/dataset/deduplication.py --dataset-dir ./datasets/sample-dataset/ \
  --threshold 0.985 \
  --batch-size 64
```

---

### Stage: `eda`

**Purpose:** Generate Exploratory Data Analysis (EDA) statistics and visualizations.

**Requires:**
- Standardized dataset

**Produces:**
- `eda_report.json` (statistics)
- Visualizations (PNG files)

**Statistics Generated:**
- Class distribution
- Image size statistics
- Bounding box statistics
- Dataset split breakdown

**Example:**
```bash
python scripts/dataset/eda_stats.py --dataset-dir ./datasets/sample-dataset/
```

---

### Stage: `train`

**Purpose:** Train YOLOv26 model on your dataset.

**Requires:**
- Prepared dataset with `data.yaml`
- Pre-trained weights (downloaded automatically if missing)

**Produces:**
- `results/FS-{name}/weights/best.pt` (best model)
- `results/FS-{name}/weights/last.pt` (last epoch)
- `results/FS-{name}/args.yaml` (training config)
- `results/FS-{name}/results.csv` (training metrics)

**Key Hyperparameters:**
- `batch_size`: Reduce if OOM; increase for better gradients
- `epochs`: More epochs = better accuracy (with early stopping)
- `imgsz`: Larger = better small object detection, but slower
- `patience`: Early stopping — stop if no improvement for N epochs

**Example:**
```bash
python scripts/training/train.py --data datasets/sample-dataset/data.yaml \
  --weights weights/yolov26n/yolo26n.pt \
  --epochs 100 \
  --batch-size 64 \
  --device 0
```

---

### Stage: `benchmark`

**Purpose:** Measure inference latency and FPS on your trained model.

**Requires:**
- Trained model (`best.pt`)
- Test dataset

**Produces:**
- Benchmark report with:
  - Average latency (ms)
  - FPS (frames/second)
  - CPU vs GPU comparisons

**Example:**
```bash
python scripts/evaluation/benchmark.py \
  --trained-weights results/FS-Pipeline-Run/weights/best.pt \
  --data datasets/sample-dataset/data.yaml \
  --warmup 10 \
  --runs 100
```

---

### Stage: `export`

**Purpose:** Convert trained PyTorch model to ONNX and TensorRT formats.

**Requires:**
- Trained model (`best.pt`)
- TensorRT (for TensorRT export)

**Produces:**
- `best.onnx` (ONNX format)
- `best.engine` (TensorRT format, GPU-specific)

**Trade-offs:**
- **PyTorch (.pt)**: Largest, slow, portable
- **ONNX (.onnx)**: Medium, faster, CPU/GPU
- **TensorRT (.engine)**: Smallest, fastest, GPU-only, hardware-specific

**Example:**
```bash
python scripts/deployment/export.py \
  --trained-weights results/FS-Pipeline-Run/weights/best.pt \
  --formats onnx tensorrt \
  --half \
  --device 0
```

---

### Stage: `inference`

**Purpose:** Run inference on video or images using the trained/exported model.

**Requires:**
- Trained or exported model
- Video file or image directory

**Produces:**
- Annotated video/images with detections
- Results saved to `results_dir`

**Split-Frame Mode:**
- Divides frame into 2×2 grid
- Runs 4 separate inferences
- Merges results
- **Benefit**: Better detection of small/distant smoke
- **Cost**: ~4× slower inference

**Standard Mode:**
```bash
python scripts/evaluation/inference.py \
  --weights results/FS-Pipeline-Run/weights/best.pt \
  --source videos/test.mp4 \
  --results-dir results/
```

**Split-Frame Mode:**
```bash
python scripts/evaluation/inference.py \
  --weights results/FS-Pipeline-Run/weights/best.pt \
  --source videos/test.mp4 \
  --results-dir results/ \
  --split-frame
```

---

## Usage Examples

### Example 1: Complete Pipeline (Default)

```bash
# Run all stages in order
python pipeline.py -c pipeline.yaml
```

**Output:**
- Clean, deduplicated dataset in `datasets/sample-dataset/`
- Trained model in `results/FS-Pipeline-Run/weights/best.pt`
- Benchmark report
- Exported models (ONNX, TensorRT)
- Inference results on test video

---

### Example 2: Dataset Preparation Only

**Goal:** Prepare and clean dataset, but don't train yet.

**Step 1:** Edit `pipeline.yaml`:
```yaml
pipeline:
  stages:
    - standardize
    - clean
    - deduplicate
    - eda
    # - train
    # - benchmark
    # - export
    # - inference
```

**Step 2:** Run:
```bash
python pipeline.py -c pipeline.yaml
```

---

### Example 3: Training & Evaluation Only

**Goal:** Train on already-prepared dataset.

**Step 1:** Edit `pipeline.yaml`:
```yaml
pipeline:
  stages:
    # - standardize
    # - clean
    # - deduplicate
    # - eda
    - train
    - benchmark
    - export
    # - inference
```

**Step 2:** Run:
```bash
python pipeline.py -c pipeline.yaml
```

---

### Example 4: Quick Test Run (Debug Mode)

**Goal:** Test pipeline with minimal compute (2 epochs).

**Step 1:** Edit `pipeline.yaml`:
```yaml
training:
  debug: true   # Only run 2 epochs
  batch_size: 16  # Smaller batch for faster iterations
```

**Step 2:** Run:
```bash
python pipeline.py -c pipeline.yaml
```

---

### Example 5: Multi-GPU Training

**Goal:** Train on multiple GPUs.

**Step 1:** Edit `pipeline.yaml`:
```yaml
training:
  device: "0,1,2,3"  # Use GPUs 0, 1, 2, 3
  batch_size: 256    # Larger batch with more GPUs
```

**Step 2:** Run:
```bash
python pipeline.py -c pipeline.yaml
```

---

### Example 6: Custom Configuration File

**Goal:** Use a different config for experiments.

**Step 1:** Copy & modify config:
```bash
cp pipeline.yaml configs/experiment-v2.yaml
# Edit configs/experiment-v2.yaml
```

**Step 2:** Run:
```bash
python pipeline.py -c configs/experiment-v2.yaml
```

---

## Advanced Topics

### Path Resolution

All paths in `pipeline.yaml` are **relative to the repository root**:

```yaml
# Correct:
dataset:
  dir: "datasets/sample-dataset/"  # ✓ Relative from repo root

training:
  weights: "weights/yolov26n/yolo26n.pt"  # ✓ Relative from repo root
```

```yaml
# Incorrect (will fail):
dataset:
  dir: "C:/Users/offic/Documents/GitHub/.../sample-dataset/"  # ✗ Absolute path
  dir: "/home/user/fire-smoke/.../sample-dataset/"  # ✗ Absolute path
```

### Multi-Experiment Management

Run multiple experiments with different configurations:

```bash
# Experiment 1: Default config
python pipeline.py -c pipeline.yaml

# Experiment 2: Larger batch size
cp pipeline.yaml configs/large-batch.yaml
# Edit: batch_size: 128
python pipeline.py -c configs/large-batch.yaml

# Experiment 3: More epochs
cp pipeline.yaml configs/long-training.yaml
# Edit: epochs: 300
python pipeline.py -c configs/long-training.yaml
```

Results are organized by experiment name:
```
results/
├── FS-Pipeline-Run/          (default)
├── FS-Large-Batch/           (experiment 2)
└── FS-Long-Training/         (experiment 3)
```

### Resuming Failed Pipeline

If the pipeline fails midway, you can resume from a specific stage:

**Option 1: Comment out completed stages**
```yaml
pipeline:
  stages:
    # - standardize    # Completed
    # - clean          # Completed
    - deduplicate    # Failed here, resume from here
    - eda
    - train
    - benchmark
    - export
    - inference
```

Then run: `python pipeline.py -c pipeline.yaml`

**Option 2: Run individual scripts**
```bash
# Resume manual execution
python scripts/dataset/deduplication.py --dataset-dir datasets/sample-dataset/
python scripts/dataset/eda_stats.py --dataset-dir datasets/sample-dataset/
python scripts/training/train.py --data datasets/sample-dataset/data.yaml ...
```

---

## Troubleshooting

### Issue: `FileNotFoundError: No such file or directory`

**Causes:**
- Paths in `pipeline.yaml` are incorrect
- Missing dataset directory
- Missing pre-trained weights

**Solutions:**
1. Verify all paths are relative to repository root
2. Check that dataset directory exists: `ls -la datasets/sample-dataset/`
3. Check that weights exist: `ls -la weights/yolov26n/`
4. Use forward slashes `/` (works on Windows, Linux, macOS)

---

### Issue: CUDA out of memory (OOM)

**Causes:**
- Batch size too large
- Image size too large
- Multi-GPU not configured properly

**Solutions:**
1. Reduce batch size:
   ```yaml
   training:
     batch_size: 32  # Reduce from 64
   ```

2. Reduce image size:
   ```yaml
   training:
     imgsz: 416  # Reduce from 640
   ```

3. Use multi-GPU:
   ```yaml
   training:
     device: "0,1,2,3"
     batch_size: 256  # Use more GPUs = larger batch
   ```

---

### Issue: Dataset not found after standardize

**Cause:**
- Script didn't complete successfully
- Paths don't match

**Solution:**
```bash
# Verify dataset structure
ls -la datasets/sample-dataset/
# Should show: data.yaml, train/, valid/, test/

# Re-run standardize with verbose output
python scripts/dataset/standardize.py --dataset-dir datasets/sample-dataset/ -v
```

---

### Issue: Training stuck or very slow

**Causes:**
- Too many workers for available CPU cores
- GPU not being used (device set to "cpu")
- Insufficient system memory

**Solutions:**
1. Check GPU usage:
   ```bash
   nvidia-smi  # Should show GPU memory usage
   ```

2. Reduce workers:
   ```yaml
   training:
     workers: 0  # Disable multi-processing
   ```

3. Verify device:
   ```yaml
   training:
     device: "0"  # Use GPU 0
   ```

---

### Issue: Pipeline asks for confirmation and hangs

**Cause:**
- `pipeline.py` waiting for user input on destructive operations

**Solution:**
- Script automatically pipes "YES" — if hanging, manually interrupt with `Ctrl+C`
- Check logs for error messages
- Run individual scripts with explicit confirmation:
  ```bash
  python scripts/dataset/clean.py --dataset-dir ... << EOF
  YES
  EOF
  ```

---

## Best Practices

1. **Backup your data**: Always backup dataset before running `clean` or `deduplicate`
2. **Use version control**: Commit `pipeline.yaml` to git to track experiments
3. **Test with debug mode**: Run with `debug: true` first (2 epochs)
4. **Monitor GPU/CPU**: Use `nvidia-smi` or `top` while training
5. **Archive results**: Save best models and results after each successful run
6. **Document hyperparameters**: Keep notes on what worked and what didn't
7. **Use meaningful names**: Make experiment names descriptive (`FS-Large-Batch-v1`, not `test`)

---

## See Also

- [README.md](../README.md) — Quick start and overview
- [docs/training-guide.md](training-guide.md) — Detailed training documentation
- [docs/dataset-preparation.md](dataset-preparation.md) — Dataset engineering guide
- [docs/deployment-guide.md](deployment-guide.md) — Export & inference guide
