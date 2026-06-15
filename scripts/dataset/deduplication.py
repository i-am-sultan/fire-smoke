import os
import sys
import gc
import logging
import argparse
import torch
import faiss
import numpy as np

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor

# =========================================================
# CONFIGURATION
# =========================================================

SPLITS = ["train", "valid", "test"]

# Default Hyperparameters
EMBEDDING_DIM = 384
K_NEIGHBORS = 20
SEARCH_CHUNK_SIZE = 8_192
ADD_CHUNK_SIZE = 50_000

# =========================================================
# LOGGING
# =========================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "deduplication.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# =========================================================
# SAFETY CONFIRMATION
# =========================================================

def _confirm_destructive_execution(dataset_dir: Path, threshold: float):
    """Halts execution to warn the user about in-place file deletion."""
    print("=" * 70)
    print(" ⚠️  WARNING: DESTRUCTIVE IN-PLACE DEDUPLICATION ⚠️")
    print("=" * 70)
    print(f"Target Directory : {dataset_dir.resolve()}")
    print(f"Similarity Thresh: {threshold} (DINOv2 + FAISS)")
    print("\nThis script will modify your dataset DIRECTLY.")
    print(" - Images identified as near-duplicates will be PERMANENTLY DELETED.")
    print(" - Their corresponding YOLO label files will be PERMANENTLY DELETED.")
    print("\nDo you have a backup of this directory? Are you sure you want to proceed?")
    
    confirm = input("Type 'YES' (all caps) to continue, or anything else to abort: ").strip()
    
    if confirm != "YES":
        print("\n[ABORTED] Safety exit triggered. No files were modified.")
        sys.exit(0)
    print("\n[PROCEEDING] Initiating AI-driven deduplication pipeline...\n")


# =========================================================
# DATASET
# =========================================================

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except Exception as e:
            logger.error(f"Failed to open {path}: {e}")
            return torch.zeros((3, 224, 224))


def load_image_paths(split_dir: Path):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_dir = split_dir / "images"

    if not img_dir.exists():
        logger.warning(f"Missing directory: {img_dir}")
        return []

    image_paths = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    return sorted(image_paths)


# =========================================================
# EXTRACT EMBEDDINGS
# =========================================================

def extract_embeddings(image_paths, device, split, batch_size):
    logger.info(f"[{split}] Loading DINOv2 model...")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    model.eval()

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    num_workers = min(4, max(1, os.cpu_count() // 2))
    dataset = ImageDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    memmap_path = f"embeddings_{split}_temp.dat"
    embeddings_memmap = np.memmap(
        memmap_path,
        dtype="float16",
        mode="w+",
        shape=(len(image_paths), EMBEDDING_DIM),
    )

    current_idx = 0
    logger.info(f"[{split}] Extracting embeddings on {device.type.upper()}")

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[{split}] Embedding Extraction"):
            batch = batch.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(pixel_values=batch)
                
            embeddings = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
            batch_size_actual = embeddings.shape[0]

            embeddings_memmap[current_idx : current_idx + batch_size_actual] = embeddings.astype("float16")
            current_idx += batch_size_actual

    embeddings_memmap.flush()
    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return embeddings_memmap, memmap_path


# =========================================================
# UNION FIND & CLUSTERING
# =========================================================

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x


def get_duplicate_paths(embeddings_memmap, image_paths, split, threshold):
    n = len(image_paths)
    k_neighbors = min(K_NEIGHBORS, n)

    faiss.omp_set_num_threads(os.cpu_count())
    index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64

    # Indexing
    for start in tqdm(range(0, n, ADD_CHUNK_SIZE), desc=f"[{split}] Indexing"):
        end = min(start + ADD_CHUNK_SIZE, n)
        chunk = embeddings_memmap[start:end].astype("float32")
        faiss.normalize_L2(chunk)
        index.add(chunk)

    uf = UnionFind(n)

    # Searching
    for start in tqdm(range(0, n, SEARCH_CHUNK_SIZE), desc=f"[{split}] Clustering"):
        end = min(start + SEARCH_CHUNK_SIZE, n)
        query_chunk = embeddings_memmap[start:end].astype("float32")
        faiss.normalize_L2(query_chunk)

        similarities, indices = index.search(query_chunk, k_neighbors)

        for local_i, (sim_row, idx_row) in enumerate(zip(similarities, indices)):
            global_i = start + local_i
            for j in range(1, k_neighbors):
                if idx_row[j] >= 0 and sim_row[j] >= threshold:
                    uf.union(global_i, idx_row[j])

    del index
    gc.collect()

    # Identify duplicates (anything that isn't the root of its cluster)
    representative_map = {}
    for idx in range(n):
        root = uf.find(idx)
        if root not in representative_map:
            representative_map[root] = idx

    kept_indices = set(representative_map.values())
    dropped_paths = [image_paths[i] for i in range(n) if i not in kept_indices]

    return dropped_paths


# =========================================================
# FILE DELETION
# =========================================================

def delete_single_duplicate(img_path: Path):
    """Deletes the image and its corresponding label file."""
    label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    
    img_path.unlink(missing_ok=True)
    label_path.unlink(missing_ok=True)
    
    return True


def purge_duplicates(dropped_paths, split):
    if not dropped_paths:
        logger.info(f"[{split}] No duplicates found! Directory is clean.")
        return

    logger.info(f"[{split}] Purging {len(dropped_paths)} duplicate images and labels...")

    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count())) as executor:
        list(tqdm(
            executor.map(delete_single_duplicate, dropped_paths),
            total=len(dropped_paths),
            desc=f"[{split}] Deleting Files"
        ))


# =========================================================
# MAIN ENTRY POINT
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "=======================================================================\n"
            " In-Place YOLO Semantic Deduplication Tool\n"
            "=======================================================================\n"
            "Uses DINOv2 and FAISS to find and delete near-duplicate images directly\n"
            "from your dataset directories.\n"
            "Example:\n\n"
            "python scripts/dataset/deduplication.py --dataset-dir /home2/testdev/sultan/fire-smoke/datasets/semi-gold/ --threshold 0.90 --batch-size 64\n"
        )
    )
    
    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="(REQUIRED) Path to the base dataset directory."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.985,
        help="Similarity threshold (0.0 to 1.0). Default is 0.985."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for DINOv2 inference. Default is 32."
    )

    args = parser.parse_args()
    base_dir = Path(args.dataset_dir)

    if not base_dir.exists():
        logger.error(f"Directory not found: {base_dir}")
        sys.exit(1)

    _confirm_destructive_execution(base_dir, args.threshold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device.type.upper()}")

    total_initial = 0
    total_dropped = 0

    for split in SPLITS:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue

        logger.info("=" * 50)
        logger.info(f"PROCESSING SPLIT: {split.upper()}")
        logger.info("=" * 50)

        image_paths = load_image_paths(split_dir)
        if not image_paths:
            logger.warning(f"[{split}] No images found. Skipping.")
            continue

        total_initial += len(image_paths)

        # 1. Extract Embeddings
        embeddings, memmap_path = extract_embeddings(
            image_paths, device, split, args.batch_size
        )

        # 2. Find Duplicates
        dropped_paths = get_duplicate_paths(
            embeddings, image_paths, split, args.threshold
        )

        # 3. Delete Duplicates
        purge_duplicates(dropped_paths, split)

        total_dropped += len(dropped_paths)

        # 4. Cleanup Memory Map
        del embeddings
        gc.collect()
        if Path(memmap_path).exists():
            os.remove(memmap_path)
            logger.info(f"[{split}] Removed temporary embeddings file.")

    total_kept = total_initial - total_dropped

    logger.info("=" * 60)
    logger.info(" IN-PLACE DEDUPLICATION REPORT")
    logger.info("=" * 60)
    logger.info(f" Initial Images : {total_initial}")
    logger.info(f" Dropped / Del  : {total_dropped}")
    logger.info(f" Remaining      : {total_kept}")
    logger.info(f" Threshold Used : {args.threshold}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()