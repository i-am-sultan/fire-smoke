import sys
import argparse
import logging
from pathlib import Path
from collections import Counter
from typing import Dict

# ==============================================================================
#  LOGGING SETUP
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 

)
logger = logging.getLogger(__name__)

# ==============================================================================
#  CORE EDA LOGIC
# ==============================================================================

def run_eda(dataset_dir: Path) -> None:
    """
    Performs a single, efficient pass over the YOLO dataset to extract:
    - File counts (images, labels, empty labels)
    - Class distributions
    - Missing pairs (images without labels, labels without images)
    """
    splits = ["train", "valid", "test"]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Initialize data structures to hold our stats
    aggregate_stats = {
        "images": 0, "labels": 0, "empty_labels": 0,
        "missing_images": 0, "missing_labels": 0,
        "classes": Counter(), "malformed_lines": 0
    }
    
    split_stats: Dict[str, dict] = {}

    logger.info("Scanning dataset. This may take a moment depending on size...\n")

    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue

        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        # 1. Collect all valid image and label stems
        image_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in exts] if images_dir.exists() else []
        label_files = [f for f in labels_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"] if labels_dir.exists() else []

        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}

        # 2. Calculate Missing Pairs
        missing_labels = len(image_stems - label_stems)
        missing_images = len(label_stems - image_stems)

        # 3. Parse Labels for Classes and Empty Files
        empty_labels = 0
        class_counts = Counter()
        malformed = 0

        for label_file in label_files:
            if label_file.stat().st_size == 0:
                empty_labels += 1
                continue
            
            try:
                lines = label_file.read_text(encoding="utf-8").splitlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    # YOLO format: class x_center y_center width height
                    if len(parts) != 5:
                        malformed += 1
                        continue
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width    = float(parts[3])
                        height   = float(parts[4])
                        class_counts[class_id] += 1
                    except ValueError:
                        malformed += 1
                        continue
            except Exception as e:
                malformed += 1

        # Store split stats
        split_stats[split] = {
            "images": len(image_files),
            "labels": len(label_files),
            "empty_labels": empty_labels,
            "missing_images": missing_images,
            "missing_labels": missing_labels,
            "classes": class_counts,
            "malformed": malformed
        }

        # Add to aggregate
        aggregate_stats["images"] += len(image_files)
        aggregate_stats["labels"] += len(label_files)
        aggregate_stats["empty_labels"] += empty_labels
        aggregate_stats["missing_images"] += missing_images
        aggregate_stats["missing_labels"] += missing_labels
        aggregate_stats["classes"].update(class_counts)
        aggregate_stats["malformed_lines"] += malformed

    _print_report(split_stats, aggregate_stats)


def _print_report(split_stats: dict, agg: dict):
    """Formats and prints the collected statistics."""
    
    logger.info("=" * 60)
    logger.info(" 📊 YOLO DATASET EDA REPORT")
    logger.info("=" * 60)

    # Print Split-level stats
    for split, stats in split_stats.items():
        logger.info(f"\n[{split.upper()}]")
        logger.info(f"  Images             : {stats['images']}")
        logger.info(f"  Labels             : {stats['labels']}")
        logger.info(f"  Empty Labels       : {stats['empty_labels']}")
        
        if stats['missing_labels'] > 0 or stats['missing_images'] > 0:
            logger.info("  -- Warnings --")
            logger.info(f"  Missing Labels     : {stats['missing_labels']} (Images with no .txt file)")
            logger.info(f"  Missing Images     : {stats['missing_images']} (.txt files with no image)")
        if stats['malformed'] > 0:
            logger.info(f"  Malformed Annot.   : {stats['malformed']} lines")

        logger.info("  -- Class Distribution --")
        if not stats['classes']:
            logger.info("    None found")
        for cid, count in sorted(stats['classes'].items()):
            logger.info(f"    Class {cid:<10}: {count}")

    # Print Aggregate stats
    logger.info("\n" + "=" * 60)
    logger.info(" 📈 AGGREGATE TOTALS")
    logger.info("=" * 60)
    logger.info(f"  Total Images       : {agg['images']}")
    logger.info(f"  Total Labels       : {agg['labels']}")
    logger.info(f"  Total Empty Labels : {agg['empty_labels']} (Backgrounds)")
    logger.info(f"  Total Missing Pairs: {agg['missing_labels'] + agg['missing_images']}")
    
    logger.info("\n  Overall Class Distribution:")
    for cid, count in sorted(agg['classes'].items()):
         logger.info(f"    Class {cid:<10}: {count}")
    
    logger.info("=" * 60 + "\n")


# ==============================================================================
#  ENTRY POINT & CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "=======================================================================\n"
            " YOLO Exploratory Data Analysis (EDA) Tool\n"
            "=======================================================================\n"
            "Analyzes a YOLO dataset directory to provide file counts, class \n"
            "distributions, and validation of missing image/label pairs.\n\n"
            "Example:\n"
            "python scripts/dataset/eda_stats.py --dataset-dir=/home2/testdev/sultan/fire-smoke/datasets/semi-gold/\n"
        )
    )
    
    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="(REQUIRED) Path to the base directory containing train/valid/test folders."
    )

    args = parser.parse_args()

    run_eda(dataset_dir=Path(args.dataset_dir))