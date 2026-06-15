import os
import sys
import time
import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ==============================================================================
#  LOGGING SETUP
# ==============================================================================
LOG_DIR = "logs"
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR,f"benchmark_{current_time}.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ==============================================================================
#  BENCHMARK FUNCTION (Pure PyTorch Inference)
# ==============================================================================

def benchmark_inference(model, device="cpu", image_size=640, warmup=10, runs=100):
    """Measures raw PyTorch network forward-pass speed (excluding pre/post-processing)."""
    logger.info(f"Benchmarking on device: {device.upper()}")

    # Extract the raw PyTorch model to avoid Ultralytics wrapper overhead
    inner_model = model.model.to(device)
    inner_model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = inner_model(dummy_input)

        if device != "cpu":
            torch.cuda.synchronize()

        # Timed runs
        start = time.time()
        for _ in range(runs):
            _ = inner_model(dummy_input)

        if device != "cpu":
            torch.cuda.synchronize()
        end = time.time()

    avg_time = (end - start) / runs
    fps = 1 / avg_time

    return avg_time * 1000, fps

# ==============================================================================
#  MAIN LOGIC
# ==============================================================================

def run_benchmark(args):
    weights_path = Path(args.trained_weights)
    if not weights_path.exists():
        logger.error(f"Model weights not found at: {weights_path}")
        sys.exit(1)

    logger.info(f"Loading model: {weights_path.name}")
    model = YOLO(weights_path)

    # -------------------------------------------------
    # 1. DATASET VALIDATION (mAP, Precision, Recall)
    # -------------------------------------------------
    if args.data:
        for data_yaml in args.data:
            yaml_path = Path(data_yaml)
            dataset_name = yaml_path.parent.name if yaml_path.exists() else "Unknown"

            logger.info("=" * 60)
            logger.info(f" 📊 EVALUATING DATASET: {dataset_name}")
            logger.info("=" * 60)

            if not yaml_path.exists():
                logger.error(f"Dataset YAML not found: {yaml_path}. Skipping.")
                continue

            # Run Ultralytics Validation
            metrics = model.val(
                data=str(yaml_path),
                split="test",
                name=f"Eval_{weights_path.stem}_{dataset_name}",
                verbose=False,
                workers=args.workers,
            )

            logger.info(f"\nFinal Stats for {dataset_name}:")
            logger.info(f" - mAP@50:      {metrics.box.map50:.4f}")
            logger.info(f" - mAP@50-95:   {metrics.box.map:.4f}")
            logger.info(f" - Precision:   {metrics.box.mp:.4f}")
            logger.info(f" - Recall:      {metrics.box.mr:.4f}")

    # -------------------------------------------------
    # 2. SPEED BENCHMARKING (CPU & GPU)
    # -------------------------------------------------
    logger.info("=" * 60)
    logger.info(" ⚡ SPEED BENCHMARKS (Raw Network Inference)")
    logger.info("=" * 60)

    # CPU Benchmark
    cpu_time_ms, cpu_fps = benchmark_inference(
        model=model, device="cpu", image_size=args.imgsz, warmup=args.warmup, runs=args.runs
    )
    logger.info(f"CPU Benchmark:")
    logger.info(f" - Avg inference time/image: {cpu_time_ms:.2f} ms")
    logger.info(f" - FPS: {cpu_fps:.2f}\n")

    # GPU Benchmark
    if torch.cuda.is_available():
        gpu_time_ms, gpu_fps = benchmark_inference(
            model=model, device="cuda:0", image_size=args.imgsz, warmup=args.warmup, runs=args.runs
        )
        logger.info(f"GPU Benchmark (CUDA:0):")
        logger.info(f" - Avg inference time/image: {gpu_time_ms:.2f} ms")
        logger.info(f" - FPS: {gpu_fps:.2f}")
    else:
        logger.warning("CUDA not available. Skipping GPU benchmark.")

    logger.info("=" * 60)
    logger.info("Benchmark completed successfully.")


# ==============================================================================
#  CLI PARSER
# ==============================================================================

if __name__ == "__main__":
    # Required for Windows multiprocessing stability
    torch.multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Standardized YOLO Benchmark Tool (Metrics & Inference Speed)"
    )
    
    # Required Arguments
    parser.add_argument("--trained-weights", type=str, required=True, help="(REQUIRED) Path to model weights (.pt file)")
    
    # Optional Arguments
    parser.add_argument("--data", type=str, nargs="+", default=[], help="One or multiple paths to dataset data.yaml files to evaluate against.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for speed benchmark (Default: 640)")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers for validation (Default: 4)")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations for speed test (Default: 10)")
    parser.add_argument("--runs", type=int, default=100, help="Number of timed iterations for speed test (Default: 100)")

    args = parser.parse_args()
    
    run_benchmark(args)