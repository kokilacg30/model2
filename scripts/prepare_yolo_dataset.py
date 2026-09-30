import os
import shutil
import random
from pathlib import Path

BASE_DIR = Path(r"e:\nitin\model2")
YOLO_DIR = BASE_DIR / "dataset" / "yolo_data"

def prepare_yolo_structure(seed=42, val_ratio=0.2):
    random.seed(seed)
    
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
        
    (YOLO_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "val" / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "val" / "labels").mkdir(parents=True, exist_ok=True)
    
    # 1. Clean Roads (Hard Negatives: Empty label files)
    clean_dir = BASE_DIR / "road cleanliness" / "clean roads"
    clean_images = list(clean_dir.glob("*.*"))
    clean_images = [img for img in clean_images if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    random.shuffle(clean_images)
    val_clean_count = int(len(clean_images) * val_ratio)
    
    for idx, img in enumerate(clean_images):
        split = "val" if idx < val_clean_count else "train"
        dest_img = YOLO_DIR / split / "images" / img.name
        dest_lbl = YOLO_DIR / split / "labels" / f"{img.stem}.txt"
        
        shutil.copy2(img, dest_img)
        # Touch empty label file for clean road
        with open(dest_lbl, "w") as f:
            pass
            
    # 2. Dirty Roads (Images with candidate trash bounding boxes)
    auto_lbl_dir = BASE_DIR / "dataset" / "auto_labeled" / "labels"
    dirty_subdirs = ["slightly dirty", "very dirty"]
    
    dirty_samples = []
    for sub in dirty_subdirs:
        sub_dir = BASE_DIR / "road cleanliness" / sub
        for img in sub_dir.glob("*.*"):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                lbl_file = auto_lbl_dir / f"{img.stem}.txt"
                dirty_samples.append((img, lbl_file))
                
    random.shuffle(dirty_samples)
    val_dirty_count = int(len(dirty_samples) * val_ratio)
    
    for idx, (img, lbl) in enumerate(dirty_samples):
        split = "val" if idx < val_dirty_count else "train"
        dest_img = YOLO_DIR / split / "images" / img.name
        dest_lbl = YOLO_DIR / split / "labels" / f"{img.stem}.txt"
        
        shutil.copy2(img, dest_img)
        if lbl.exists():
            shutil.copy2(lbl, dest_lbl)
        else:
            with open(dest_lbl, "w") as f:
                pass

    # Create dataset.yaml
    yaml_content = f"""path: {YOLO_DIR.as_posix()}
train: train/images
val: val/images

names:
  0: trash
"""
    yaml_file = YOLO_DIR / "dataset.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
        
    print(f"YOLO dataset preparation complete! Structure saved to: {YOLO_DIR}")
    print(f"Dataset YAML: {yaml_file}")

if __name__ == "__main__":
    prepare_yolo_structure()
