import sys
import argparse
import logging
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# ==============================================================================
#  LOGGING SETUP
# ==============================================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "standardize.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
#  SAFETY CONFIRMATION
# ==============================================================================

def _confirm_destructive_execution(dataset_dir: Path):
    """Halts execution to warn the user about in-place file modification."""
    print("=" * 70)
    print(" ⚠️  WARNING: DESTRUCTIVE IN-PLACE STANDARDIZATION ⚠️")
    print("=" * 70)
    print(f"Target Directory: {dataset_dir.resolve()}")
    print("\nThis script will modify your dataset DIRECTLY.")
    print(" - Annotations will be overwritten based on your mapping.")
    print(" - Unmapped classes will be DELETED from labels.")
    print(" - Orphaned label files (no matching image) will be DELETED.")
    print(" - Blank text files will be created for background images.")
    print("\nDo you have a backup of this directory? Are you sure you want to proceed?")
    
    confirm = input("Type 'YES' (all caps) to continue, or anything else to abort: ").strip()
    
    if confirm != "YES":
        print("\n[ABORTED] Safety exit triggered. No files were modified.")
        sys.exit(0)
    print("\n[PROCEEDING] Initiating in-place standardization pipeline...\n")

# ==============================================================================
#  WORKER FUNCTION
# ==============================================================================

def process_single_item(img_path: Path, label_path: Path, mapping: dict):
    """
    Worker function to process a single image/label pair.
    Reads, remaps, and overwrites the label file in-place.
    """
    local_stats = {
        "objects_mapped": 0,
        "unmapped_objects_dropped": 0,
        "is_background": True,
        "was_missing_label": False
    }

    if not label_path.exists():
        # Create empty label file for background images
        label_path.touch()
        local_stats["was_missing_label"] = True
        return {"status": "success", "stats": local_stats}

    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            try:
                old_class_id = int(parts[0])
            except ValueError:
                continue # Skip malformed lines
            
            # If mapping is empty, we keep the original ID. Otherwise, we enforce the map.
            if not mapping:
                new_lines.append(line.strip() + "\n")
                local_stats["is_background"] = False
                local_stats["objects_mapped"] += 1
            elif old_class_id in mapping:
                new_class_id = mapping[old_class_id]
                new_lines.append(f"{new_class_id} " + " ".join(parts[1:]) + "\n")
                local_stats["is_background"] = False
                local_stats["objects_mapped"] += 1
            else:
                local_stats["unmapped_objects_dropped"] += 1

        # Overwrite the file in-place
        with open(label_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    except Exception as e:
        return {"status": "error", "error_msg": f"Label processing failed {label_path.name}: {e}"}

    return {"status": "success", "stats": local_stats}

# ==============================================================================
#  MAIN LOGIC
# ==============================================================================

def run_standardization(base_dir: Path, mapping: dict, class_names: list):
    splits = ["train", "valid", "test", "val"] # Added 'val' as it's common in YOLO
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    stats = {
        "total_images": 0, "background_images": 0, 
        "total_objects_mapped": 0, "unmapped_objects_dropped": 0, 
        "blank_labels_generated": 0, "orphaned_labels_deleted": 0,
        "failed_files": 0
    }

    tasks_to_run = []

    logger.info("Scanning directories and preparing tasks...")

    for split in splits:
        images_path = base_dir / split / "images"
        labels_path = base_dir / split / "labels"

        if not images_path.exists():
            continue

        # Ensure labels directory exists
        labels_path.mkdir(parents=True, exist_ok=True)

        # 1. Gather all valid images
        image_stems = set()
        for img_path in images_path.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                image_stems.add(img_path.stem)
                label_path = labels_path / f"{img_path.stem}.txt"
                tasks_to_run.append((img_path, label_path))
                stats["total_images"] += 1

        # 2. Find and delete orphaned labels (labels with no matching image)
        for label_path in labels_path.iterdir():
            if label_path.is_file() and label_path.suffix == ".txt":
                if label_path.stem not in image_stems:
                    label_path.unlink()
                    stats["orphaned_labels_deleted"] += 1

    total_tasks = len(tasks_to_run)
    if total_tasks == 0:
        logger.error("No valid images found in the specified directory.")
        return

    logger.info(f"Starting in-place modification of {total_tasks} files...")

    # ThreadPool for fast Disk I/O
    with ThreadPoolExecutor() as executor:
        future_to_task = {
            executor.submit(process_single_item, img_path, lbl_path, mapping): lbl_path 
            for img_path, lbl_path in tasks_to_run
        }
        
        for i, future in enumerate(as_completed(future_to_task), 1):
            try:
                result = future.result()
                if result["status"] == "error":
                    logger.error(result["error_msg"])
                    stats["failed_files"] += 1
                else:
                    loc_stats = result["stats"]
                    stats["total_objects_mapped"] += loc_stats["objects_mapped"]
                    stats["unmapped_objects_dropped"] += loc_stats["unmapped_objects_dropped"]
                    
                    if loc_stats["is_background"]:
                        stats["background_images"] += 1
                    if loc_stats["was_missing_label"]:
                        stats["blank_labels_generated"] += 1

            except Exception as exc:
                logger.error(f"Task generated an unhandled exception: {exc}")
                stats["failed_files"] += 1
                
            if i % 2000 == 0:
                logger.info(f"Progress: {i}/{total_tasks} files processed...")

    # ==========================================
    # DATA.YAML GENERATION
    # ==========================================
    if not class_names:
        # Default fallback if user provides no classes
        class_names = [f"class_{i}" for i in range(max(mapping.values()) + 1 if mapping else 1)]

    yaml_data = {
        'path': str(base_dir.resolve()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    try:
        with open(base_dir / "data.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, sort_keys=False)
        logger.info("Generated clean data.yaml")
    except Exception as e:
        logger.error(f"Failed to save data.yaml: {e}")

    # ==========================================
    # FINAL REPORT
    # ==========================================
    logger.info("="*50)
    logger.info("🎯 DATASET STANDARDIZATION COMPLETE")
    logger.info("="*50)
    logger.info(f" Total Images Evaluated   : {stats['total_images']}")
    logger.info(f" Background Images (Empty): {stats['background_images']}")
    logger.info(f" Blank Labels Generated   : {stats['blank_labels_generated']}")
    logger.info(f" Orphaned Labels Deleted  : {stats['orphaned_labels_deleted']}")
    logger.info(f" Total Objects Mapped     : {stats['total_objects_mapped']}")
    logger.info(f" Unmapped Objects Dropped : {stats['unmapped_objects_dropped']}")
    if stats["failed_files"] > 0:
        logger.warning(f" Failed Files Skipped     : {stats['failed_files']}")
    logger.info("="*50)


# ==============================================================================
#  CLI PARSER
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "=======================================================================\n"
            " YOLO In-Place Dataset Standardizer\n"
            "=======================================================================\n"
            "Cleans a downloaded dataset by optionally remapping class IDs, dropping\n"
            "unwanted classes, fixing missing labels, and generating a data.yaml.\n\n"
            "EXAMPLES:\n"
            "  1. Just fix missing labels and generate YAML (No remapping):\n"
            "     python standardize_yolo.py --dataset-dir data/my_dataset --classes fire smoke\n\n"
            "  2. Remap classes (Old 0 -> New 0, Old 1 -> New 2, Old 2 -> New 1):\n"
            "     python standardize_yolo.py --dataset-dir data/my_dataset \\\n"
            "         --map 0:0 1:2 2:1 --classes fire smoke other\n"
        )
    )
    
    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="(REQUIRED) Path to the root of the dataset containing train/valid folders."
    )
    parser.add_argument(
        "--map", type=str, nargs="+", default=[],
        help="(OPTIONAL) Remap rules in 'old:new' format (e.g., --map 0:0 1:2 2:1)."
    )
    parser.add_argument(
        "--classes", type=str, nargs="+", required=True,
        help="(REQUIRED) Final class names for data.yaml (e.g., --classes fire smoke other)."
    )

    args = parser.parse_args()
    target_dir = Path(args.dataset_dir)

    if not target_dir.exists():
        logger.error(f"Directory not found: {target_dir}")
        sys.exit(1)

    # Parse mapping strings (e.g., "0:0", "1:2") into a dictionary {0: 0, 1: 2}
    mapping_dict = {}
    try:
        for rule in args.map:
            old_id, new_id = rule.split(":")
            mapping_dict[int(old_id)] = int(new_id)
    except ValueError:
        logger.error("Invalid --map format. Must be 'old:new' integers (e.g., 0:1 2:0).")
        sys.exit(1)

    _confirm_destructive_execution(target_dir)

    run_standardization(target_dir, mapping_dict, args.classes)