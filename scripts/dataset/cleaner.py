import os
import sys
import hashlib
import random
import logging
import argparse
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
#  CONFIGURATION & CONSTANTS
# ==============================================================================

SPLITS = ["train", "valid", "test"]

# Cleaning thresholds
MIN_WIDTH              = 150
MIN_HEIGHT             = 150
MAX_ASPECT_RATIO       = 5.0   # Drop images that are excessively long/skinny

# Curation settings
TARGET_BACKGROUND_RATIO = 0.30 # 30% background images is standard for YOLO
FIRE_SMOKE_CLASS_IDS    = (0, 1) # class 0 = fire, class 1 = smoke; class 2 = other (dropped)

# ==============================================================================
#  LOGGING SETUP
# ==============================================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "cleaner.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ==============================================================================
#  SAFETY CONFIRMATION
# ==============================================================================

def _confirm_destructive_execution(dataset_dir: Path):
    """Halts execution to warn the user about in-place file deletion."""
    print("=" * 70)
    print(" ⚠️  WARNING: DESTRUCTIVE IN-PLACE OPERATION ⚠️")
    print("=" * 70)
    print(f"Target Directory: {dataset_dir.resolve()}")
    print("\nThis script will modify your dataset DIRECTLY.")
    print(" - Corrupt files and duplicates will be DELETED.")
    print(" - Annotations for 'class 2' will be ERASED.")
    print(" - Excess background images will be DELETED.")
    print("\nDo you have a backup of this directory? Are you sure you want to proceed?")
    
    confirm = input("Type 'YES' (all caps) to continue, or anything else to abort: ").strip()
    
    if confirm != "YES":
        print("\n[ABORTED] Safety exit triggered. No files were modified.")
        sys.exit(0)
    print("\n[PROCEEDING] Initiating in-place data pipeline...\n")

# ==============================================================================
#  STAGE 1: IN-PLACE CLEANING (Corruptions, Sizes, Duplicates)
# ==============================================================================

def _analyze_image(img_path: Path) -> dict:
    """Reads one image, computes MD5 hash, and validates integrity."""
    result = {
        "path":       img_path,
        "is_valid":   False,
        "reason":     "",
        "hash":       None,
    }

    # 0. Zero-byte guard
    if img_path.stat().st_size == 0:
        result["reason"] = "Zero-byte file"
        return result

    # 1. MD5 hash (for exact duplicate detection)
    try:
        result["hash"] = hashlib.md5(img_path.read_bytes()).hexdigest()
    except Exception as exc:
        result["reason"] = f"File Read Error: {exc}"
        return result

    # 2. Integrity & dimension check
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            result["reason"] = "Corrupt/Unreadable Image"
            return result

        h, w, _ = img.shape
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            result["reason"] = f"Too Small ({w}x{h})"
            return result

        if max(w / h, h / w) > MAX_ASPECT_RATIO:
            result["reason"] = f"Extreme Aspect Ratio ({max(w/h, h/w):.2f})"
            return result

    except Exception as exc:
        result["reason"] = f"Decode Error: {exc}"
        return result

    result["is_valid"] = True
    result["reason"]   = "Passed"
    return result


def run_inplace_cleaning(base_dir: Path) -> bool:
    """Scans dataset, deletes invalid images and their labels."""
    image_paths = []
    
    for split in SPLITS:
        split_img_dir = base_dir / split / "images"
        if split_img_dir.exists():
            image_paths.extend([f for f in split_img_dir.iterdir() if f.is_file()])
            
    total = len(image_paths)
    if total == 0:
        logger.warning(f"No images found in {base_dir}. Skipping cleaning.")
        return False

    logger.info(f"[STAGE 1] Analysing {total} images for integrity & duplicates...")
    analysis_results = []

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_analyze_image, p): p for p in image_paths}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                analysis_results.append(future.result())
            except Exception as exc:
                logger.error(f"Analysis task failed: {exc}")
            if i % 2000 == 0:
                logger.info(f"  Analysed {i}/{total} ...")

    # Sort deterministically to keep the first lexicographical file when dropping duplicates
    analysis_results.sort(key=lambda r: r["path"].name)

    seen_hashes  = set()
    stats = {"passed": 0, "dropped_corrupt": 0, "dropped_size": 0, 
             "dropped_duplicate": 0, "dropped_zero_byte": 0}

    logger.info(f"[STAGE 1] Deleting invalid files in-place...")

    for res in analysis_results:
        img_path = res["path"]
        label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"

        # Determine if we should drop it
        drop = False
        if not res["is_valid"]:
            drop = True
            if "Zero-byte" in res["reason"]: stats["dropped_zero_byte"] += 1
            elif "Small" in res["reason"] or "Aspect" in res["reason"]: stats["dropped_size"] += 1
            else: stats["dropped_corrupt"] += 1
        elif res["hash"] in seen_hashes:
            drop = True
            stats["dropped_duplicate"] += 1
        else:
            seen_hashes.add(res["hash"])
            stats["passed"] += 1

        # Perform destructive action if invalid
        if drop:
            img_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)

    logger.info("=" * 45)
    logger.info("🧹 STAGE 1: IN-PLACE CLEANING AUDIT")
    logger.info("=" * 45)
    logger.info(f" Total images checked           : {total}")
    logger.info(f" ✅ Kept valid images           : {stats['passed']}")
    logger.info(" ❌ DELETED from disk:")
    logger.info(f"    Exact duplicates            : {stats['dropped_duplicate']}")
    logger.info(f"    Too small / bad aspect ratio: {stats['dropped_size']}")
    logger.info(f"    Corrupt / unreadable        : {stats['dropped_corrupt']}")
    logger.info(f"    Zero-byte files             : {stats['dropped_zero_byte']}")
    logger.info("=" * 45)

    return stats["passed"] > 0


# ==============================================================================
#  STAGE 2: IN-PLACE CURATION (Class Filtering & Background Trimming)
# ==============================================================================

def run_inplace_curation(base_dir: Path):
    """Strips class-2 annotations and deletes excess background images."""
    image_paths = []
    
    for split in SPLITS:
        split_img_dir = base_dir / split / "images"
        if split_img_dir.exists():
            image_paths.extend(list(split_img_dir.glob("*.*")))

    annotated   = []   # Contains target classes
    backgrounds = []   # Empty labels or stripped to empty
    dropped_class2 = 0

    logger.info(f"[STAGE 2] Curating annotations for {len(image_paths)} images...")

    for img_path in image_paths:
        label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        
        valid_lines = []
        has_fire_or_smoke = False

        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                
                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue
                
                if class_id in FIRE_SMOKE_CLASS_IDS:
                    valid_lines.append(line + "\n")
                    has_fire_or_smoke = True
                elif class_id == 2:
                    dropped_class2 += 1

            # Overwrite the label file with only valid classes
            label_path.write_text("".join(valid_lines))
        else:
            # Create an empty text file for backgrounds
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("")

        if has_fire_or_smoke:
            annotated.append(img_path)
        else:
            backgrounds.append(img_path)

    total_annotated = len(annotated)
    total_backgrounds_initial = len(backgrounds)

    # ------------------------------------------------------------------
    #  Enforce 30% Background Ratio
    # ------------------------------------------------------------------
    # Math: target_bgs / (total_annotated + target_bgs) = 0.20
    # Therefore: target_bgs = total_annotated * 0.20 / 0.90
    target_bg_count = int(total_annotated * (TARGET_BACKGROUND_RATIO / (1.0 - TARGET_BACKGROUND_RATIO)))
    
    # Shuffle deterministically to randomly select which backgrounds to drop
    backgrounds.sort() 
    random.seed(42)
    random.shuffle(backgrounds)

    discarded_bgs = 0
    kept_bgs = total_backgrounds_initial

    if len(backgrounds) > target_bg_count:
        excess_backgrounds = backgrounds[target_bg_count:]
        kept_bgs = target_bg_count
        discarded_bgs = len(excess_backgrounds)
        
        logger.info(f"[STAGE 2] Deleting {discarded_bgs} excess background images...")
        for bg_img in excess_backgrounds:
            bg_label = bg_img.parent.parent / "labels" / f"{bg_img.stem}.txt"
            bg_img.unlink(missing_ok=True)
            bg_label.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Rewrite data.yaml safely
    # ------------------------------------------------------------------
    yaml_path = base_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {base_dir.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 2\n"
        "names: ['fire', 'smoke']\n"
    )

    logger.info("=" * 45)
    logger.info("🥇 STAGE 2: DATASET CURATION COMPLETE")
    logger.info("=" * 45)
    logger.info(f" Annotated images (fire/smoke)  : {total_annotated}")
    logger.info(f" Background images kept (30%)   : {kept_bgs}")
    logger.info(f" Total images remaining         : {total_annotated + kept_bgs}")
    logger.info(f" 'Class 2' annotations deleted  : {dropped_class2}")
    if discarded_bgs > 0:
        logger.info(f" Excess backgrounds DELETED     : {discarded_bgs}")
    logger.info("=" * 45)


# ==============================================================================
#  ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "=======================================================================\n"
            " In-Place YOLO Dataset Cleaner & Curator\n"
            "=======================================================================\n"
            "Modifies a dataset directory DIRECTLY by removing corruptions, \n"
            "deleting duplicates, stripping unwanted classes, and trimming backgrounds.\n\n"
            "EXAMPLES:\n"
            "  python cleaner.py --dataset-dir D:\\FIRE-SMOKE-DATASET\n"
        )
    )
    
    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="(REQUIRED) Path to the base directory containing train/valid/test folders."
    )
    
    args = parser.parse_args()
    target_dir = Path(args.dataset_dir)
    
    if not target_dir.exists():
        logger.error(f"Directory not found: {target_dir}")
        sys.exit(1)

    _confirm_destructive_execution(target_dir)

    if run_inplace_cleaning(target_dir):
        run_inplace_curation(target_dir)
    else:
        logger.error("Pipeline halted: cleaning stage produced no valid images.")