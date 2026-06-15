import os
import sys
import time
import cv2
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
        logging.FileHandler(os.path.join(LOG_DIR,f"inference_{current_time}.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ==============================================================================
#  HARDWARE DETECTOR
# ==============================================================================
def get_optimal_device(requested_device: str):
    """Dynamically detects and assigns the best available hardware engine."""
    if requested_device.lower() != "auto":
        if requested_device.lower() == "cpu":
            return "cpu"
        return int(requested_device) if requested_device.isdigit() else requested_device

    if not torch.cuda.is_available():
        logger.warning("No CUDA GPUs detected. Defaulting execution engine to CPU.")
        return "cpu"
    
    logger.info("GPU hardware accelerator detected. Routing execution to CUDA:0.")
    return 0

# ==============================================================================
#  PIPELINE 1: STANDARD FULL FRAME INFERENCE
# ==============================================================================
def run_standard_inference(model, args, video_path, output_dir):
    """Executes standard YOLO inference on a full, unsliced video stream."""
    logger.info("Executing standard full-frame tracking pipeline...")
    
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_time = time.time()
    results = model.predict(
        source=str(video_path),
        device=get_optimal_device(args.device),
        imgsz=args.imgsz,
        conf=args.conf,
        verbose=False,
        save=True,
        project=str(output_dir),
        name=f"{video_path.stem}_standard",
        exist_ok=True,
    )
    end_time = time.time()
    
    elapsed = end_time - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    
    logger.info(f"Standard pipeline completed in {elapsed:.2f} seconds.")
    logger.info(f"Processed Frames: {frame_count} | Effective Processing Rate: {fps:.2f} FPS")
    logger.info(f"Inference outputs safely stored in: {output_dir / f'{video_path.stem}_standard'}")

# ==============================================================================
#  PIPELINE 2: QUADRANT SPLIT-AND-MERGE INFERENCE
# ==============================================================================
def run_quadrant_inference(model, args, video_path, output_dir):
    """Splits frames into quadrants, processes them, and stitches them back together."""
    logger.info("⚠️ Split-frame mode active: Commencing 4-quadrant slicing operations...")
    
    # 1. Initialize original video stream properties
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to access source video stream at: {video_path}")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    mid_x, mid_y = frame_width // 2, frame_height // 2

    # Temp directory to hold transient quadrant chunks
    temp_dir = output_dir / f"{video_path.stem}_quadrant_cache"
    temp_dir.mkdir(parents=True, exist_ok=True)

    quad_paths = [temp_dir / f"q{i}.mp4" for i in range(1, 5)]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = [cv2.VideoWriter(str(p), fourcc, fps_video, (mid_x, mid_y)) for p in quad_paths]

    logger.info("Step 1: Slicing parent video stream into spatial quadrant streams...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Crop slices
        writers.write(frame[0:mid_y, 0:mid_x])                # Top-Left
        writers.write(frame[0:mid_y, mid_x:frame_width])       # Top-Right
        writers.write(frame[mid_y:frame_height, 0:mid_x])       # Bottom-Left
        writers.write(frame[mid_y:frame_height, mid_x:frame_width]) # Bottom-Right

    cap.release()
    for w in writers:
        w.release()

    # 2. Run inference across the independent quadrants
    logger.info("Step 2: Dispatching localized neural network predictions across quadrants...")
    device_id = get_optimal_device(args.device)
    
    for idx, q_path in enumerate(quad_paths):
        logger.info(f" -> Processing Quadrant {idx+1}/4 ({q_path.name})...")
        model.predict(
            source=str(q_path),
            device=device_id,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            save=True,
            project=str(temp_dir),
            name=f"processed_q{idx+1}",
            exist_ok=True,
        )

    # 3. Stitch quadrants back into unified resolution layout
    logger.info("Step 3: Compiling processed outputs back into a unified matrix layout...")
    
    # Ultralytics natively saves predicted clips as .avi extensions
    processed_avi_paths = [temp_dir / f"processed_q{i}" / f"q{i}.avi" for i in range(1, 5)]
    caps = [cv2.VideoCapture(str(v)) for v in processed_avi_paths]

    # Validate stream setups
    for i, c in enumerate(caps):
        if not c.isOpened():
            logger.error(f"Failed to access generated sub-inference stream: {processed_avi_paths[i]}")
            sys.exit(1)

    final_output_path = output_dir / f"{video_path.stem}_quadrant_stitched.avi"
    out_fourcc = cv2.VideoWriter_fourcc(*"XVID")
    merged_writer = cv2.VideoWriter(str(final_output_path), out_fourcc, fps_video, (frame_width, frame_height))

    frame_idx = 0
    while True:
        rets_frames = [c.read() for c in caps]
        if not all(ret for ret, _ in rets_frames):
            break

        frames = [cv2.resize(f, (mid_x, mid_y)) for _, f in rets_frames]
        
        # Concat rows
        top_row = cv2.hconcat([frames, frames])
        bottom_row = cv2.hconcat([frames, frames])
        
        # Merge rows horizontally/vertically
        merged_frame = cv2.vconcat([top_row, bottom_row])
        merged_writer.write(merged_frame)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            logger.info(f" Progress: Compiled {frame_idx}/{frame_count} frames back to matrix array.")

    for c in caps:
        c.release()
    merged_writer.release()
    
    logger.info(f"✓ Pipeline successful. Complete split-frame grid saved at:\n{final_output_path}")

# ==============================================================================
#  MAIN SYSTEM CONTROLLER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
                "Unified Enterprise Inference Execution Pipeline - Standard vs. Quadrant Slicing Modalities.\n"
                "Example:\n"
                'python scripts/evaluation/inference.py --trained-weights "/home2/testdev/sultan/fire-smoke-training-pipeline/results/FS-TRAIN01-2/weights/best.pt" --source "/home2/testdev/sultan/fire-smoke/datasets/videos/samples/part1/bucket11.mp4" --results-dir /home2/testdev/sultan/fire-smoke-training-pipeline/results/\n'
        )
    )
    
    # Core parameters
    parser.add_argument("--trained-weights", type=str, required=True, help="(REQUIRED) Path to object detection model file (.pt, .onnx, or .engine)")
    parser.add_argument("--source", type=str, required=True, help="(REQUIRED) Path to input video file target (.mp4, .avi)")
    parser.add_argument("--results-dir", type=str, required=True, help="(REQUIRED) Destination directory for target data footprints")
    
    # Configuration parameters
    parser.add_argument("--imgsz", type=int, default=640, help="Internal image dimensions parameter (Default: 640)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence filter threshold cutoff metric (Default: 0.5)")
    parser.add_argument("--device", type=str, default="auto", help="Compute execution engine routing targets: 'cpu', '0', 'auto'")
    
    # The split frame toggle flag
    parser.add_argument("--split-frame", action="store_true", 
                        help="Toggles 4-quadrant slicing layout. Splits frame 2x2, scales small inference zones, and stitches back.")

    args = parser.parse_args()

    # Dynamic Path Validations
    video_path = Path(args.source).resolve()
    output_dir = Path(args.results_dir).resolve()
    
    if not video_path.exists():
        logger.error(f"Declared system source target invalid: {video_path}")
        sys.exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing objection detection model instance from {Path(args.trained_weights).name}...")
    model = YOLO(args.trained_weights, task="detect")

    # Routing core workflow conditional branches
    if args.split_frame:
        run_quadrant_inference(model, args, video_path, output_dir)
    else:
        run_standard_inference(model, args, video_path, output_dir)


if __name__ == "__main__":
    main()