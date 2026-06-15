# 🔥 Fire & Smoke Detection Training & Deployment Pipeline

An end-to-end dataset engineering, quality assurance, training, evaluation, and deployment framework for **Fire & Smoke Detection using YOLOv26**.

This repository provides a complete workflow for transforming raw datasets into deployment-ready fire and smoke detection models through automated dataset preparation, semantic deduplication, model training, benchmarking, and hardware-optimized deployment.

> [!NOTE]
> This repository does **not** include datasets or trained weights.
> Users are expected to prepare their own datasets and train models using the provided pipeline. Refer to the documentation under `docs/` for the complete workflow.

---

## 🎯 Repository Philosophy

Most computer vision repositories focus primarily on model training. This repository emphasizes the complete machine learning lifecycle:

1. Dataset Quality
2. Dataset Diversity
3. Reproducible Experiments
4. Deployment Readiness

The goal is to help users build robust fire and smoke detection systems using their own datasets rather than relying on pre-packaged data.

---

## 🚀 Core Features

### 📊 Dataset Engineering & Quality Assurance

* Dataset standardization and class remapping
* Automated YAML generation
* Background image support
* Annotation validation and visualization
* Corruption detection and removal
* Extreme aspect ratio filtering
* Tiny object filtering
* Background balancing

**Semantic Deduplication**
Near-duplicate image removal powered by DINOv2 feature extraction, FAISS similarity search, and HNSW indexing. This significantly improves dataset diversity while reducing training redundancy.

### 🏋️ Model Training

* YOLOv26 training pipeline
* Automatic CPU / GPU detection (Single-GPU and Multi-GPU support)
* Resume training support
* Deterministic experiment execution
* Domain-specific augmentation strategies
* Automated dataset integrity verification

### 📈 Evaluation & Benchmarking

* mAP@50 and mAP@50-95
* Precision & Recall
* CPU & GPU latency benchmarking
* FPS measurement
* Cross-dataset evaluation support

### 📦 Deployment

* PyTorch deployment, ONNX export, and TensorRT export
* Standard video inference
* Quadrant split-frame inference
* Production deployment validation

**Split-Frame Inference:** A specialized deployment mode that divides frames into a `2×2` grid before inference and merges the results afterward. This yields improved small smoke detection, better distant-object detection, and higher effective inference resolution.

---

## 🤖 Supported Models

| Model | Purpose |
| --- | --- |
| **YOLOv26n** | Lightweight deployment and faster inference |
| **YOLOv26s** | Higher accuracy deployment |

*Additional YOLOv26 variants can be integrated through the training pipeline.*

---

## 🚀 Getting Started

### Prerequisites

* **Python:** 3.9+
* **Hardware:** GPU with CUDA support and the latest drivers installed
* **Knowledge:** Basic understanding of Exploratory Data Analysis (EDA)
* **Data:** Your dataset must follow the standard YOLO annotation format

---

### 1. Clone Repository

```bash
git clone http://10.10.30.65:3000/trainee-ai-ml/fire-smoke-training-pipeline.git
cd fire-smoke-training-pipeline

```

### 2. Create Virtual Environment

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate

```

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1

```

### 3. Install Dependencies

```bash
pip install tensorrt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

```

> **Note:** Install cuda compiled torch, torchvision and torchaudio based on your cuda version only.

---

## 📚 Documentation

| Guide | Description |
| --- | --- |
| [architecture.md](docs/architecture.md) | System architecture and data flow |
| [dataset-preparation.md](docs/dataset-preparation.md) | Dataset engineering and validation |
| [training-guide.md](docs/training-guide.md) | Training workflow and hyperparameters |
| [evaluation-guide.md](docs/evaluation-guide.md) | Performance analysis and benchmarking |
| [deployment-guide.md](docs/deployment-guide.md) | Export and deployment workflows |
| [troubleshooting.md](docs/troubleshooting.md) | FAQ and issue resolution |

---

## 🛠️ Standard Workflow

> [!TIP]
> Run any script with the `-h` or `--help` argument to view a complete list of available features and parameters.

### Step 1 — Dataset Preparation

📖 **Documentation:** [docs/dataset-preparation.md](docs/dataset-preparation.md)

Prepare, standardize, clean, and validate your dataset.

```bash
python scripts/dataset/standardize.py --dataset-dir ./datasets/sample-dataset/

python scripts/dataset/cleaner.py --dataset-dir ./datasets/sample-dataset/

python scripts/dataset/deduplication.py --dataset-dir ./datasets/sample-dataset/ --threshold 0.9 --batch-size 64

```

Before proceeding to training, verify your dataset quality and class distributions:

```bash
python scripts/dataset/eda_stats.py --dataset-dir ./datasets/sample-dataset/

```

### Step 2 — Train Model

📖 **Documentation:** [docs/training-guide.md](docs/training-guide.md)

```bash
python scripts/training/train.py --weights ./weights/yolov26n/yolov26n.pt --data ./datasets/sample-dataset/data.yaml --imgsz 640 --epochs 100 --device 0 --results-dir ./results --name YOLOv26n-v1 --debug
```

### Step 3 — Evaluate Performance

📖 **Documentation:** [docs/evaluation-guide.md](docs/evaluation-guide.md)

> [!IMPORTANT]
> Before running this script, ensure the paths defined in your `data.yaml` correctly match the location of your finalized dataset.

```bash
python scripts/evaluation/benchmark.py --trained-weights .\results\FS-YOLOv26n-v1\weights\best.pt --data datasets\sample-dataset\data.yaml

```

### Step 4 — Export Deployment Artifacts

📖 **Documentation:** [docs/deployment-guide.md](docs/deployment-guide.md)

```bash
python scripts/deployment/export.py --trained-weights results/yolov26n/best.pt --formats onnx tensorrt --imgsz 640 --half --device 0
```

### Step 5 — Run Inference

📖 **Documentation:** [docs/deployment-guide.md](docs/deployment-guide.md)

**Standard Inference**

```bash
python scripts/evaluation/inference.py --weights models/yolov26n/best.engine --source test.mp4 --results-dir result/

```

**Split-Frame Inference**

```bash
python scripts/evaluation/inference.py --weights models/yolov26n/best.engine --source test.mp4 --results-dir result/ --split-frame

```

---

## 🏗️ Pipeline Overview

```text
Raw Dataset
      │
      ▼
Standardization ──► Cleaning ──► Semantic Deduplication ──► Dataset Validation
                                                                  │
                                                                  ▼
Inference ◄── Export ◄── Benchmarking ◄── Training ◄──────────────┘

```

---

## 📂 Repository Structure

```text
fire-smoke-training-pipeline/
├── docs/
│   ├── architecture.md
│   ├── dataset-preparation.md
│   ├── training-guide.md
│   ├── evaluation-guide.md
│   ├── deployment-guide.md
│   └── troubleshooting.md
├── scripts/
│   ├── dataset/
│   ├── training/
│   ├── evaluation/
│   └── deployment/
├── models/
├── logs/
├── results/
├── requirements.txt
└── README.md

```

---

## 📋 Best Practices

* **Prioritize recall** for fire-safety applications to minimize missed detections.
* **Validate annotations** visually before initiating large-scale training.
* **Remove semantic duplicates** to ensure the model learns diverse features rather than memorizing redundant frames.
* **Benchmark** every exported model to ensure latency requirements are met.
* **Archive** experiment logs and benchmark reports for traceability.
* **Build TensorRT engines natively** on the exact target hardware to avoid compatibility issues.

---

## 📈 Results

Benchmark results and deployment metrics should be recorded and versioned for each experiment. Example directory structure:

```text
results/
├── FS-YOLOv26n-v1/
├── FS-YOLOv26n-v2/
├── FS-YOLOv26s-v1/
└── benchmarks/

```

---

## 🤝 Contributing

Contributions are welcome. Areas of interest include:

* Dataset engineering
* Training improvements
* Benchmarking enhancements
* Deployment optimizations
* Documentation improvements
* CI/CD integration

---

## 🙏 Acknowledgements

Built using: **YOLOv26**, **PyTorch**, **Ultralytics**, **DINOv2**, **FAISS**, and **OpenCV**.

---

## 📌 Final Note

High-performing fire and smoke detection systems are built on high-quality datasets. This repository prioritizes dataset quality, reproducible experimentation, and deployment readiness to help teams build reliable, real-world solutions.