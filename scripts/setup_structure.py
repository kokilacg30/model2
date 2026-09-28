import os
import shutil
from pathlib import Path

BASE_DIR = Path(r"e:\nitin\model2")
DATASET_DIR = BASE_DIR / "dataset"

# Define directory structure
dirs = [
    DATASET_DIR / "raw" / "clean_roads",
    DATASET_DIR / "raw" / "slightly_dirty",
    DATASET_DIR / "raw" / "very_dirty",
    DATASET_DIR / "annotation",
    DATASET_DIR / "images",
    DATASET_DIR / "labels",
    DATASET_DIR / "train" / "images",
    DATASET_DIR / "train" / "labels",
    DATASET_DIR / "val" / "images",
    DATASET_DIR / "val" / "labels",
    DATASET_DIR / "test" / "images",
    DATASET_DIR / "test" / "labels",
    DATASET_DIR / "hard_negatives",
    BASE_DIR / "results",
    BASE_DIR / "docs",
    BASE_DIR / "scripts"
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)
    print(f"Created/Verified directory: {d}")

# Copy raw images into dataset/raw structure
src_raw = BASE_DIR / "road cleanliness"
category_map = {
    "clean roads": DATASET_DIR / "raw" / "clean_roads",
    "slightly dirty": DATASET_DIR / "raw" / "slightly_dirty",
    "very dirty": DATASET_DIR / "raw" / "very_dirty",
}

copied_count = 0
for cat, dst_dir in category_map.items():
    cat_src = src_raw / cat
    if cat_src.exists():
        for img_file in cat_src.glob("*.*"):
            dst_file = dst_dir / img_file.name
            if not dst_file.exists():
                shutil.copy2(img_file, dst_file)
            copied_count += 1

print(f"Structure setup completed. Copied/synced {copied_count} files into dataset/raw/")
