# 🛠️ Troubleshooting & FAQ Guide

## Overview

This document covers the most common issues encountered while preparing datasets, training models, evaluating results, deploying systems, and managing Python environments.

Before opening an issue or escalating a ticket, please review this guide.

---

## 📦 Environment Setup & Dependencies

**Q: Installation fails while building `pandas`, `scipy`, or other scientific packages.**

* **Symptoms:** Error messages like `Building wheel for scipy`, `OpenBLAS not found`, or `subprocess-exited-with-error`.
* **Explanation:** `pip` attempts to build packages from source when a compatible precompiled wheel (`.whl`) is unavailable for your OS or Python version. Building these from source requires C/Fortran compilers and system libraries.
* **Solution:** Upgrade your installation tools first: `pip install --upgrade pip setuptools wheel`. Ensure you are using a Python version that has official wheels available (typically Python 3.9 - 3.11).

**Q: Dependency resolution fails with `ResolutionImpossible`.**

* **Explanation:** Two packages require incompatible versions of the same underlying dependency (e.g., Package A requires `numpy < 2.0`, but Package B requires `numpy >= 2.0`).
* **Solution:** Stick strictly to the provided `requirements.txt`. Do not manually upgrade scientific packages unless necessary.

**Q: PyTorch installed successfully, but CUDA is not available.**

* **Symptoms:** `torch.cuda.is_available()` returns `False`.
* **Explanation:** You likely installed the CPU-only version of PyTorch, or your NVIDIA drivers are mismatched with the installed PyTorch CUDA toolkit.
* **Solution:** Verify your GPU is visible using `nvidia-smi`. Ensure you install PyTorch using the specific index URL provided in the main README (e.g., `--index-url https://download.pytorch.org/whl/cu121`).

**Q: Should I use Conda, `venv`, or both?**

* **Solution:** Use **one** environment manager. Do not nest a `venv` inside an active Conda environment, as this introduces severe path confusion and dependency inconsistencies. On standard Linux/Windows, use `venv`. On HPC nodes (like Kshitij), use Conda.

---

## 📊 Dataset Preparation

**Q: The training script says "Dataset YAML not found".**

* **Solution:** Verify that your `data.yaml` actually exists at the path you specified. Double-check your command line arguments for typos.

**Q: Training starts but reports zero images found.**

* **Solution:** Verify that your `data.yaml` correctly points to the `images/` directory, not just the root `train/` folder. Ensure your directory structure exactly matches the standard YOLO format.

**Q: Why are some images missing after running `cleaner.py`?**

* **Explanation:** The cleaner automatically removes corrupt images, zero-byte files, extreme aspect ratios, and exact duplicates. This is expected behavior. Always check the generated log file to see exactly what was removed.

**Q: Why did `cleaner.py` remove my background images?**

* **Explanation:** The pipeline maintains a target background ratio. Excess negative samples are pruned to keep the dataset balanced and prevent the model from biasing toward predicting "nothing".

**Q: Why do I have completely empty `.txt` label files?**

* **Explanation:** Empty `.txt` files represent background images (images with no fire or smoke). This is the correct, expected YOLO format. **Do not delete them.**

**Q: I have duplicates remaining after running `cleaner.py`.**

* **Explanation:** The cleaner only removes *exact* byte-for-byte duplicates. CCTV footage often contains hundreds of near-identical sequential frames.
* **Solution:** Run the Semantic Deduplication script (`deduplication.py`) to remove visually similar frames using DINOv2 embeddings.

---

## 🏋️ Training & GPU Usage

**Q: Training crashes with `CUDA out of memory`.**

* **Solution:** Your GPU VRAM is full. Reduce your `--batch-size` (e.g., from 64 to 32) or decrease your input resolution (`--imgsz`).

**Q: Training is extremely slow.**

* **Possible Causes:** You are accidentally training on the CPU, reading data from a slow HDD, or using an excessive number of data loader workers.
* **Solution:** Ensure you pass `--device auto`, verify your data is on an SSD, and lower `--workers` if CPU bottlenecks occur.

**Q: Why is training stuck at 0%?**

* **Solution:** This is often a data-loader deadlock. Try setting `--workers 0` in your training command. If it starts successfully, the issue is related to multi-processing on your specific OS.

**Q: Can I stop training and continue later?**

* **Solution:** Yes. Point your training script to the `last.pt` weights file and pass the `--resume` flag.

---

## 📈 Evaluation & Metrics

**Q: My mAP is very low. What should I check?**

* **Solution:** Almost all low-mAP issues stem from the dataset. Verify your class mappings (`0=Fire`, `1=Smoke`), ensure annotations are normalized YOLO coordinates, and run `visualize.py` to physically look at your bounding boxes.

**Q: Training mAP is high, but validation mAP is low.**

* **Explanation:** Your model is overfitting (memorizing the training data).
* **Solution:** Add more diverse data, include more background (empty) images, and run semantic deduplication to force the model to learn general features rather than identical frames.

**Q: Precision is high, but Recall is low.**

* **Explanation:** The model is too conservative. It rarely triggers false alarms, but it frequently misses actual fires.
* **Solution:** In fire safety, **Recall is paramount**. To fix this, you may need more diverse fire examples or a lower confidence threshold during inference.

---

## 📦 Export & Deployment

**Q: TensorRT export failed.**

* **Solution:** Verify that the TensorRT Python bindings are installed (`pip install tensorrt`). Ensure your CUDA version matches the installed TensorRT libraries.

**Q: Can I build a TensorRT engine on my laptop and deploy it on the server?**

* **Solution:** Generally, no. TensorRT engines (`.engine` files) are heavily optimized for the specific GPU architecture they are compiled on. Always build the engine directly on the target deployment machine.

**Q: Real-time inference is slower than the benchmark script reported.**

* **Explanation:** The `benchmark.py` script measures *raw model inference* using synthetic tensors. Real-world video inference (`inference.py`) includes overhead from video decoding (OpenCV), frame resizing, Non-Maximum Suppression (NMS), and rendering.

**Q: Small smoke regions are not being detected in deployment.**

* **Solution:** Try exporting your model at a higher resolution (e.g., `--imgsz 1280`) or utilize **Quadrant Inference Mode** (`--split-frame`) to process high-resolution patches independently.

---

## 🐛 Submitting a Bug Report

If you have validated your dataset, reviewed the logs, and are still encountering an issue, please gather the following information before opening a ticket:

1. **Environment:** OS, Python version, PyTorch version, CUDA version, and GPU model.
2. **Command:** The exact command you ran.
3. **Traceback:** The complete error output from the terminal.
4. **Reproducibility:** A minimal set of steps required to reproduce the error.