import os
import sys
import logging
import argparse
from pathlib import Path
from ultralytics import YOLO

# ==============================================================================
#  LOGGING SETUP
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ==============================================================================
#  EXPORT CONTROLLER
# ==============================================================================
def run_export(args):
    weights_path = Path(args.trained_weights).resolve()
    if not weights_path.exists():
        logger.error(f"Target PyTorch weights not found at: {weights_path}")
        sys.exit(1)

    logger.info(f"Loading PyTorch model baseline: {weights_path.name}")
    model = YOLO(weights_path)

    # Determine execution device mapping
    # Note: TensorRT MUST be compiled on the exact target GPU environment
    device = int(args.device) if args.device.isdigit() else args.device

    for fmt in args.formats:
        fmt_lower = fmt.strip().lower()
        
        # Normalize naming conventions ('tensorrt' maps to 'engine')
        target_format = "engine" if fmt_lower in ["tensorrt", "engine", "trt"] else "onnx"
        
        logger.info("\n" + "=" * 60)
        logger.info(f" 📦 COMMENCING DEPLOYMENT EXPORT: {target_format.upper()}")
        logger.info("=" * 60)

        try:
            # Trigger Ultralytics export pipeline
            exported_path = model.export(
                format=target_format,
                imgsz=args.imgsz,
                dynamic=args.dynamic,
                simplify=args.simplify,
                opset=args.opset,
                half=args.half,
                device=device,
                int8=args.int8,
            )
            logger.info(f"✓ Successfully compiled and saved export to:\n{exported_path}")
            
        except Exception as e:
            logger.error(f"Critical failure exporting to format [{target_format}]: {e}")
            if target_format == "engine":
                logger.warning(
                    "TensorRT compilation requires explicit CUDA driver setups and matching TensorRT system libraries."
                )
            sys.exit(1) 

    logger.info("\n" + "=" * 60)
    logger.info("Export automation pipeline completed.")

# ==============================================================================
#  CLI PARSER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Unified Production Export Utility for ONNX and TensorRT Runtimes.\n"
            "Example:\n"
            'python scripts/deployment/export.py --trained-weights "/home2/testdev/sultan/fire-smoke-training-pipeline/results/FS-TRAIN01-2/weights/best.pt" --half\n'
        )
    )
    
    # Required Arguments
    parser.add_argument("--trained-weights", type=str, required=True, help="(REQUIRED) Path to input PyTorch weights file (.pt)")
    
    # Export Format Selection
    parser.add_argument("--formats", type=str, nargs="+", default=["onnx", "tensorrt"],
                        help="Export format list targets. Choices: 'onnx', 'tensorrt' (Default: both)")
    
    # Optimization Parameters
    parser.add_argument("--imgsz", type=int, default=640, help="Fixed network input resolution shape (Default: 640)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX operator set version level (Default: 17)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index mapping for TRT engine compilation (Default: '0')")
    
    # Performance Flags
    parser.add_argument("--half", action="store_true", help="Enables FP16 half-precision optimization (Crucial for TensorRT speed)")
    parser.add_argument("--int8", action="store_true", help="Enables INT8 post-training quantization")
    parser.add_argument("--dynamic", action="store_true", help="Enables dynamic shape input variables (Disables certain TRT layer optimizations)")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false", help="Disables ONNX graph simplification")
    
    parser.set_defaults(simplify=True)
    args = parser.parse_args()
    
    run_export(args)