import os
import sys
from pathlib import Path
import cv2
from PIL import Image

def validate_split(split_name, base_dir):
    img_dir = base_dir / split_name / "images"
    lbl_dir = base_dir / split_name / "labels"
    
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[{split_name.upper()}] Directory missing: {img_dir} or {lbl_dir}")
        return False
        
    img_files = {f.stem: f for f in img_dir.glob("*.*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
    lbl_files = {f.stem: f for f in lbl_dir.glob("*.txt")}
    
    print(f"\n--- VALIDATING SPLIT: {split_name.upper()} ---")
    print(f"Total Images: {len(img_files)}")
    print(f"Total Label Files: {len(lbl_files)}")
    
    missing_labels = set(img_files.keys()) - set(lbl_files.keys())
    orphan_labels = set(lbl_files.keys()) - set(img_files.keys())
    
    if missing_labels:
        print(f"  [WARNING] {len(missing_labels)} images missing label files (may be unannotated or clean negatives).")
    if orphan_labels:
        print(f"  [ERROR] {len(orphan_labels)} label files have no corresponding image file!")
        
    valid_labels_count = 0
    total_objects = 0
    corrupt_images = 0
    formatting_errors = 0
    
    for stem, img_path in img_files.items():
        # Check image integrity
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            print(f"  [CORRUPT IMAGE] {img_path.name}: {e}")
            corrupt_images += 1
            
        # Check label content if exists
        lbl_path = lbl_files.get(stem)
        if lbl_path:
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                
            for idx, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) < 5:
                    print(f"  [LABEL ERROR] {lbl_path.name} line {idx}: Bounding box / Polygon must have >= 5 values (class x1 y1 x2 y2 ...)")
                    formatting_errors += 1
                    continue
                    
                cls_id = parts[0]
                if cls_id != "0":
                    print(f"  [LABEL ERROR] {lbl_path.name} line {idx}: Class ID must be 0 (found '{cls_id}')")
                    formatting_errors += 1
                    
                coords = [float(p) for p in parts[1:]]
                if any(c < 0.0 or c > 1.0 for c in coords):
                    print(f"  [LABEL ERROR] {lbl_path.name} line {idx}: Normalized coordinates must be between 0 and 1.")
                    formatting_errors += 1
                    
                if len(parts[1:]) % 2 != 0:
                    print(f"  [LABEL ERROR] {lbl_path.name} line {idx}: Polygon coordinates must be even count of (x, y) pairs.")
                    formatting_errors += 1
                    
                total_objects += 1
            valid_labels_count += 1

    print(f"Summary for {split_name}:")
    print(f"  Corrupt Images: {corrupt_images}")
    print(f"  Formatting Errors: {formatting_errors}")
    print(f"  Total Annotated Objects (Class 0 'trash'): {total_objects}")
    return corrupt_images == 0 and formatting_errors == 0

def main():
    base_dir = Path(r"e:\nitin\model2\dataset")
    print("==================================================")
    print("STREET AIQ - DATASET ANNOTATION VALIDATION TOOL")
    print("==================================================")
    
    splits = ["train", "val", "test"]
    all_ok = True
    for s in splits:
        ok = validate_split(s, base_dir)
        if not ok:
            all_ok = False
            
    print("\n==================================================")
    if all_ok:
        print("Dataset structure and labels passed all validation checks!")
    else:
        print("Validation finished with warnings/errors. Review output above.")
    print("==================================================")

if __name__ == "__main__":
    main()
