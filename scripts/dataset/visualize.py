import cv2
import math
import random
import logging
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Predefined colors (BGR format for OpenCV)
CLASS_COLORS = [
    (0, 0, 255),    # Red
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
]

def draw_yolo_boxes(
    image: np.ndarray, 
    label_path: Path, 
    class_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Draw YOLO bounding boxes on an image.
    
    YOLO format expected in text file:
    <class_id> <x_center> <y_center> <width> <height>
    (all values normalized between 0 and 1)
    """
    if not label_path.exists():
        return image

    h, w = image.shape[:2]

    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        logger.warning(f"Failed to read label file {label_path.name}: {e}")
        return image

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        try:
            class_id_float, x_center, y_center, bw, bh = map(float, parts)
            class_id = int(class_id_float)
        except ValueError:
            logger.warning(f"Malformed line in {label_path.name}: {line}")
            continue

        # Convert normalized coordinates to pixel coordinates
        x_center *= w
        y_center *= h
        bw *= w
        bh *= h

        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)

        # Determine color and label
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        
        if class_names and class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label background and text for better visibility
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(
            image, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

    return image


def show_random_samples_grid(
    image_dir: Path,
    label_dir: Path,
    class_names: Optional[List[str]] = None,
    num_samples: int = 10,
    filename_contains: Optional[str] = None
) -> None:
    """
    Randomly select images and display them in a grid with YOLO bounding boxes.
    """
    logger.info(f"Scanning directories...\nImages: {image_dir}\nLabels: {label_dir}")

    if not image_dir.exists() or not label_dir.exists():
        logger.error("Image or label directory does not exist. Please check your paths.")
        return

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    valid_pairs = []

    # Iterate through labels and find matching images
    for label_path in label_dir.glob("*.txt"):
        base_name = label_path.stem
        
        # Apply filename filter if provided
        if filename_contains and filename_contains not in base_name:
            continue

        # Look for a matching image file
        image_path = None
        for ext in exts:
            candidate = image_dir / f"{base_name}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path:
            valid_pairs.append((image_path, label_path))

    if not valid_pairs:
        logger.error("No matching image-label pairs found. Check your extensions or directories.")
        return

    logger.info(f"Found {len(valid_pairs)} valid image-label pairs.")

    # Randomly sample
    num_samples = min(num_samples, len(valid_pairs))
    samples = random.sample(valid_pairs, num_samples)

    # Calculate grid size
    cols = min(5, num_samples)
    rows = math.ceil(num_samples / cols)

    plt.figure(figsize=(4 * cols, 4 * rows))

    for idx, (img_path, lbl_path) in enumerate(samples):
        image = cv2.imread(str(img_path))

        if image is None:
            logger.warning(f"Could not load image: {img_path.name}")
            continue

        # Draw boxes and convert color space for Matplotlib
        image = draw_yolo_boxes(image, lbl_path, class_names)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Plotting
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(image_rgb)
        
        # Truncate title if too long to prevent layout breaking
        title = img_path.name
        if len(title) > 30:
            title = title[:27] + "..."
            
        plt.title(title, fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    logger.info("Rendering plot...")
    plt.show()


import sys # <-- Make sure to add this at the top of your file with the other imports

if __name__ == "__main__":
    # 1. Use RawTextHelpFormatter to allow multi-line descriptions and examples
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "=======================================================================\n"
            " YOLO Annotation Visualizer\n"
            "=======================================================================\n"
            "Visualize YOLO dataset annotations on a random sample of images.\n\n"
            "EXAMPLES:\n"
            "  Basic (Required only):\n"
            "    python visualize.py --image-dir data/images --label-dir data/labels\n\n"
            "  Advanced (With optionals):\n"
            "    python visualize.py --image-dir data/images --label-dir data/labels --classes fire smoke --num-samples 15\n"
        )
    )
    
    parser.add_argument(
        "--image-dir", type=str, required=True,
        help="(REQUIRED) Path to the directory containing images."
    )
    parser.add_argument(
        "--label-dir", type=str, required=True,
        help="(REQUIRED) Path to the directory containing YOLO text labels."
    )
    parser.add_argument(
        "--classes", type=str, nargs="+", default=[],
        help="(OPTIONAL) Space-separated list of class names.\n[Default: Empty (Will just show Class IDs)]"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="(OPTIONAL) Number of random samples to display.\n[Default: 10]"
    )
    parser.add_argument(
        "--contains", type=str, default=None,
        help="(OPTIONAL) Only process files whose names contain this substring.\n[Default: None]"
    )

    # Parse arguments
    args = parser.parse_args()

    # 2. Check if the user ONLY passed the required arguments
    # We do this by checking if any of the optional flags exist in the command they typed
    optional_flags = {"--classes", "--num-samples", "--contains"}
    used_flags = set(sys.argv)
    
    if not optional_flags.intersection(used_flags):
        print("\n[INFO] You are running with only the required arguments.")
        print("The script will use the following default settings:")
        print(f"  -> Classes:      None (Using raw Class IDs)")
        print(f"  -> Num Samples:  {args.num_samples}")
        print(f"  -> Filter:       {args.contains}")
        
        # 3. Ask for confirmation
        confirm = input("\nDo you want to proceed with these defaults? [Y/n]: ").strip().lower()
        if confirm not in ['', 'y', 'yes']:
            print("Aborted by user.")
            sys.exit(0)
        print("Proceeding...\n")

    # Run the main function
    show_random_samples_grid(
        image_dir=Path(args.image_dir),
        label_dir=Path(args.label_dir),
        class_names=args.classes,
        num_samples=args.num_samples,
        filename_contains=args.contains,
    )