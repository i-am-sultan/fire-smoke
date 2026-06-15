import os
import sys
import yaml
import logging
import argparse
import subprocess
from pathlib import Path

# ==============================================================================
#  LOGGING SETUP
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PipelineMaster")

# Map stages to their respective script paths
SCRIPT_MAP = {
    "standardize": "scripts/dataset/standardize.py",
    "clean": "scripts/dataset/cleaner.py",
    "deduplicate": "scripts/dataset/deduplication.py",
    "eda": "scripts/dataset/eda_stats.py",
    "train": "scripts/training/train.py",
    "benchmark": "scripts/evaluation/benchmark.py",
    "export": "scripts/deployment/export.py",
    "inference": "scripts/evaluation/inference.py"
}

# ==============================================================================
#  CORE EXECUTION LOGIC
# ==============================================================================
def run_command(cmd: list, require_confirmation: bool = False):
    """Executes a CLI command and streams the output."""
    cmd_str = " ".join([str(c) for c in cmd])
    logger.info(f"🚀 Executing: {cmd_str}")
    
    try:
        if require_confirmation:
            # Automatically pipe 'YES' to bypass the safety prompts in dataset scripts
            process = subprocess.run(
                [str(c) for c in cmd], 
                input="YES\n", 
                text=True, 
                check=True
            )
        else:
            process = subprocess.run(
                [str(c) for c in cmd], 
                check=True
            )
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Stage failed with exit code {e.returncode}. Aborting pipeline.")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"❌ Could not find the script at: {cmd[1]}")
        sys.exit(1)

# ==============================================================================
#  STAGE BUILDERS
# ==============================================================================
def run_pipeline(config_path: Path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    stages = config.get("pipeline", {}).get("stages", [])
    if not stages:
        logger.warning("No stages defined in the pipeline configuration.")
        return

    logger.info("=" * 60)
    logger.info(f" ⚙️  INITIALIZING PIPELINE CONFIG: {config_path.name}")
    logger.info("=" * 60)

    for stage in stages:
        if stage not in SCRIPT_MAP:
            logger.error(f"Unknown stage '{stage}'. Skipping.")
            continue
            
        script_path = Path(SCRIPT_MAP[stage])
        cmd = [sys.executable, str(script_path)]
        needs_confirm = False

        logger.info("\n" + "=" * 60)
        logger.info(f" ➡️ STARTING STAGE: {stage.upper()}")
        logger.info("=" * 60)

        # ---------------------------------------------------------
        # DATASET STAGES
        # ---------------------------------------------------------
        if stage in ["standardize", "clean", "deduplicate", "eda"]:
            dataset_cfg = config.get("dataset", {})
            dataset_dir = dataset_cfg.get("dir")
            
            if not dataset_dir:
                logger.error("Dataset directory not specified in config. Aborting.")
                sys.exit(1)
                
            cmd.extend(["--dataset-dir", dataset_dir])
            
            if stage == "standardize":
                needs_confirm = True
                cmd.extend(["--classes"] + dataset_cfg.get("classes", []))
                if dataset_cfg.get("map"):
                    cmd.extend(["--map"] + dataset_cfg.get("map"))
            elif stage == "clean":
                needs_confirm = True
            elif stage == "deduplicate":
                needs_confirm = True
                cmd.extend([
                    "--threshold", str(dataset_cfg.get("dedup_threshold", 0.985)),
                    "--batch-size", str(dataset_cfg.get("dedup_batch_size", 32))
                ])

        # ---------------------------------------------------------
        # TRAINING STAGE
        # ---------------------------------------------------------
        elif stage == "train":
            train_cfg = config.get("training", {})
            cmd.extend([
                "--data", train_cfg.get("data_yaml"),
                "--weights", train_cfg.get("weights"),
                "--results-dir", train_cfg.get("results_dir"),
                "--name", train_cfg.get("name"),
                "--epochs", str(train_cfg.get("epochs", 150)),
                "--patience", str(train_cfg.get("patience", 20)),
                "--imgsz", str(train_cfg.get("imgsz", 640)),
                "--batch-size", str(train_cfg.get("batch_size", 16)),
                "--workers", str(train_cfg.get("workers", 8)),
                "--device", str(train_cfg.get("device", "auto"))
            ])
            if train_cfg.get("debug"):
                cmd.append("--debug")

        # ---------------------------------------------------------
        # BENCHMARK STAGE
        # ---------------------------------------------------------
        elif stage == "benchmark":
            bench_cfg = config.get("benchmark", {})
            train_cfg = config.get("training", {})
            
            # Predict the location of the best.pt from the training stage
            best_weights = Path(train_cfg.get("results_dir")) / f"FS-{train_cfg.get('name')}" / "weights" / "best.pt"
            
            cmd.extend([
                "--trained-weights", str(best_weights),
                "--data", train_cfg.get("data_yaml"),
                "--imgsz", str(train_cfg.get("imgsz", 640)),
                "--warmup", str(bench_cfg.get("warmup", 10)),
                "--runs", str(bench_cfg.get("runs", 100))
            ])

        # ---------------------------------------------------------
        # EXPORT STAGE
        # ---------------------------------------------------------
        elif stage == "export":
            export_cfg = config.get("export", {})
            train_cfg = config.get("training", {})
            best_weights = Path(train_cfg.get("results_dir")) / f"FS-{train_cfg.get('name')}" / "weights" / "best.pt"
            
            cmd.extend([
                "--trained-weights", str(best_weights),
                "--formats"
            ] + export_cfg.get("formats", ["onnx", "tensorrt"]))
            
            cmd.extend([
                "--imgsz", str(train_cfg.get("imgsz", 640)),
                "--opset", str(export_cfg.get("opset", 17)),
                # CHANGE THIS LINE: Don't inherit 'auto' from train_cfg
                "--device", str(export_cfg.get("device", "0")) 
            ])
            
            if export_cfg.get("half"): cmd.append("--half")
            if export_cfg.get("int8"): cmd.append("--int8")
            if export_cfg.get("dynamic"): cmd.append("--dynamic")
            
        # ---------------------------------------------------------
        # INFERENCE STAGE
        # ---------------------------------------------------------
        elif stage == "inference":
            inf_cfg = config.get("inference", {})
            train_cfg = config.get("training", {})
            export_cfg = config.get("export", {})
            
            # Determine which weights to use (default to engine if exported, else pt)
            results_dir = Path(train_cfg.get("results_dir"))
            weights_dir = results_dir / f"FS-{train_cfg.get('name')}" / "weights"
            
            if "tensorrt" in export_cfg.get("formats", []) or "engine" in export_cfg.get("formats", []):
                weights_path = weights_dir / "best.engine"
            else:
                weights_path = weights_dir / "best.pt"
                
            cmd.extend([
                "--trained-weights", str(weights_path),
                "--source", inf_cfg.get("source"),
                "--results-dir", str(results_dir),
                "--imgsz", str(train_cfg.get("imgsz", 640)),
                "--conf", str(inf_cfg.get("conf", 0.5)),
                "--device", str(train_cfg.get("device", "auto"))
            ])
            
            if inf_cfg.get("split_frame"):
                cmd.append("--split-frame")

        # Execute the built command
        run_command(cmd, require_confirmation=needs_confirm)

    logger.info("\n" + "=" * 60)
    logger.info(" 🎉 ENTIRE PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)

# ==============================================================================
#  ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YAML-driven YOLO Pipeline Orchestrator")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        default="configs/pipeline.yaml",
        help="Path to the pipeline configuration YAML file."
    )
    
    args = parser.parse_args()
    config_file = Path(args.config)
    
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_file.resolve()}")
        sys.exit(1)
        
    run_pipeline(config_file)