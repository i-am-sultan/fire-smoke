import os
import sys
import yaml
import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO, settings

# ==============================================================================
#  LOGGING SETUP (Auto-creates directory)
# ==============================================================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"train_{current_time}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ==============================================================================
#  ORG-STANDARD AUGMENTATIONS (Physics-based for Fire/Smoke)
# ==============================================================================
BASE_AUGMENTATIONS = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10.0,         # Rotates images with random angle (+10 to -10)
    "mosaic": 0.5,           # Stitches 4 images together (0.5 = 50% probability)
    "mixup": 0.01,            # Blends two images together
    "rect": False,           # Handle 358x640 efficiently (False for Multi-GPU)
    "close_mosaic": 10       # Train on real, un-mosaiced images for last 10 epochs
}

# ==============================================================================
#  HELPERS
# ==============================================================================

def get_optimal_device(requested_device: str):
    """Dynamically detects and assigns the best available hardware."""
    if requested_device.lower() != "auto":
        # Respect explicit user overrides
        if requested_device.lower() == "cpu":
            return "cpu"
        return [int(x.strip()) for x in requested_device.split(",")] if "," in requested_device else [int(requested_device)]

    # Auto-detect hardware
    if not torch.cuda.is_available():
        logger.warning("No GPUs detected. Falling back to CPU training.")
        return "cpu"

    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        logger.info("1 GPU detected. Assigning to device 0.")
        return 0
    else:
        gpu_list = list(range(num_gpus))
        logger.info(f"{num_gpus} GPUs detected. Assigning Multi-GPU: {gpu_list}")
        return gpu_list

def print_dataset_metadata(yaml_path: Path):
    """Logs the count of images in each split to verify dataset integrity."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        logger.info("="*50)
        logger.info(" 📊 DATASET INVENTORY")
        logger.info("="*50)

        base_path = yaml_path.parent

        for split in ['train', 'val', 'test']:
            total_images = 0
            folders = data.get(split, [])
            
            if isinstance(folders, str): 
                folders = [folders]
                
            for folder in folders:
                full_path = (base_path / folder).resolve()
                if full_path.exists():
                    count = sum(1 for f in full_path.iterdir() if f.suffix.lower() in {'.jpg', '.png', '.jpeg'})
                    logger.info(f" [{split.upper()}] {folder}: {count} images")
                    total_images += count
                else:
                    logger.warning(f" [{split.upper()}] PATH MISSING: {full_path}")
            
            logger.info(f" Total {split} images: {total_images}\n")
        logger.info("="*50)
    except Exception as e:
        logger.error(f"Failed to read metadata: {e}")
        
# ==============================================================================
#  CORE PIPELINE
# ==============================================================================

def run_training_pipeline(args):
    """Executes the YOLO training and validation workflow."""
    yaml_path = Path(args.data)
    if not yaml_path.exists():
        logger.error(f"Dataset YAML not found at: {yaml_path}")
        sys.exit(1)
        
    print_dataset_metadata(yaml_path)

    logger.info(f"{'='*60}\n 🚀 RUNNING YOLO EXPERIMENT: {args.name}\n{'='*60}")
    
    if args.debug:
        logger.warning("!!! DEBUG MODE ACTIVE: Running fast, minimal epochs !!!")

    is_resume = args.weights.endswith("last.pt") or args.resume
    if is_resume:
        logger.info(f"Resume flag triggered. Resuming training from {args.weights}...")
    else:
        logger.info(f"Starting fresh training run from {args.weights}...")

    # Determine hardware dynamically
    active_device = get_optimal_device(args.device)

    # --- KEY FIX: Resolve absolute path and override global settings ---
    results_dir_abs = Path(args.results_dir).resolve()
    
    # Override global Ultralytics settings to prevent prepending "runs/detect/"
    settings.update({
        "runs_dir": str(results_dir_abs)
    })

    try:
        model = YOLO(args.weights)
        model.train(
            data=str(yaml_path),
            project=str(results_dir_abs),    # Use absolute path here
            name=f"FS-{args.name}",
            resume=is_resume,
            seed=42,
            deterministic=True,
            epochs=2 if args.debug else args.epochs,
            patience=1 if args.debug else args.patience,
            imgsz=160 if args.debug else args.imgsz,
            fraction=0.01 if args.debug else 1.0,
            batch=16 if args.debug else args.batch_size,
            device=active_device,
            workers=4 if args.debug else args.workers,
            optimizer=args.optimizer,
            half=True,
            cache=False,
            plots=True,
            **BASE_AUGMENTATIONS
        )

        # --- KEY FIX: Log the actual save directory to verify ---
        logger.info(f"Trainer save_dir = {model.trainer.save_dir}")

        logger.info(f"Training complete. Evaluating {args.name} on Test Split...")
        
        # Test split evaluation
        metrics = model.val(
            split='test',
            project=str(results_dir_abs),    # Use absolute path here           
            name=f"FS-{args.name}_Test_Eval"
        )

        logger.info("\n" + "="*50)
        logger.info(f" 🏆 FINAL METRICS FOR {args.name}:")
        logger.info("="*50)
        logger.info(f" - mAP@50:    {metrics.box.map50:.4f}")
        logger.info(f" - mAP@50-95: {metrics.box.map:.4f}")
        logger.info(f" - Recall:    {metrics.box.mr:.4f} (Crucial for fire safety)")
        logger.info(f" - Precision: {metrics.box.mp:.4f} (Crucial for false alarms)")
        logger.info("="*50)

    except Exception as e:
        logger.error(f"Critical error during {args.name}: {e}")
        sys.exit(1)

# ==============================================================================
#  CLI PARSER
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Standardized YOLO Training Pipeline for Fire/Smoke Detection."
    )
    
    # Required Arguments
    parser.add_argument("--data", type=str, required=True, help="(REQUIRED) Path to dataset data.yaml")
    parser.add_argument("--results-dir", type=str, required=True, help="(REQUIRED) Base directory to save experiment results")
    parser.add_argument("--weights", type=str, required=True, help="(REQUIRED) Path to starting weights")
    parser.add_argument("--name", type=str, required=True, help="(REQUIRED) Experiment name (e.g., Standard-Nano)")
    
    # Optional Hyperparameters
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs (Default: 150)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (Default: 20)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (Default: 640)")
    parser.add_argument("--batch-size", type=int, default=192, help="Batch size (Default: 192)")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers (Default: 8)")
    parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer (Default: auto)")
    
    # Updated Device Argument (Defaults to 'auto')
    parser.add_argument("--device", type=str, default="auto", help="CUDA device IDs or 'auto' (Default: 'auto')")
    
    # Flags
    parser.add_argument("--debug", action="store_true", help="Run a rapid 2-epoch test on 1%% of data to verify pipeline.")
    parser.add_argument("--resume", action="store_true", help="Force resume from checkpoint")

    args = parser.parse_args()
    
    run_training_pipeline(args)