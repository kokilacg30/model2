import os
import glob
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

RAW_DIR = Path(r"e:\nitin\model2\road cleanliness")
RESULTS_DIR = Path(r"e:\nitin\model2\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def compute_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def compute_dhash(image, hash_size=8):
    # Resize and convert to grayscale for perceptual hashing
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    diff = gray[:, 1:] > gray[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def dhash_distance(h1, h2):
    return bin(h1 ^ h2).count('1')

def scan_dataset():
    categories = ['clean roads', 'slightly dirty', 'very dirty']
    records = []
    
    md5_dict = {}
    dhash_list = []
    
    print("Starting raw dataset analysis...")
    
    for cat in categories:
        cat_dir = RAW_DIR / cat
        if not cat_dir.exists():
            print(f"Directory {cat_dir} does not exist!")
            continue
            
        files = sorted(list(cat_dir.glob("*.*")))
        print(f"Processing category '{cat}': found {len(files)} files.")
        
        for file_path in files:
            ext = file_path.suffix.lower()
            rel_path = file_path.relative_to(RAW_DIR)
            
            # File size
            size_bytes = file_path.stat().st_size
            
            # Check corrupt / readable
            corrupted = False
            error_msg = ""
            width, height = 0, 0
            laplacian_var = 0.0
            mean_brightness = 0.0
            std_brightness = 0.0
            dhash_val = None
            md5_val = ""
            
            try:
                # PIL read check
                with Image.open(file_path) as img:
                    img.verify()
                
                # Re-open for numpy/cv2 processing
                img_cv = cv2.imread(str(file_path))
                if img_cv is None:
                    corrupted = True
                    error_msg = "OpenCV failed to read image"
                else:
                    height, width, channels = img_cv.shape
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    mean_brightness = float(np.mean(gray))
                    std_brightness = float(np.std(gray))
                    
                    md5_val = compute_md5(file_path)
                    dhash_val = compute_dhash(img_cv)
                    
            except Exception as e:
                corrupted = True
                error_msg = str(e)
                
            record = {
                "filename": file_path.name,
                "category": cat,
                "path": str(file_path),
                "rel_path": str(rel_path),
                "ext": ext,
                "size_kb": round(size_bytes / 1024.0, 2),
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 3) if height > 0 else 0,
                "corrupted": corrupted,
                "error_msg": error_msg,
                "laplacian_var": round(laplacian_var, 2),
                "mean_brightness": round(mean_brightness, 2),
                "std_brightness": round(std_brightness, 2),
                "md5": md5_val,
                "dhash": dhash_val
            }
            
            records.append(record)
            
            if md5_val:
                md5_dict.setdefault(md5_val, []).append(record["rel_path"])
            if dhash_val is not None:
                dhash_list.append((record["rel_path"], dhash_val))
                
    # Detect exact duplicates (MD5)
    exact_duplicates = {k: v for k, v in md5_dict.items() if len(v) > 1}
    
    # Detect perceptual near-duplicates (dhash hamming distance <= 2)
    near_duplicates = []
    num_records = len(dhash_list)
    for i in range(num_records):
        for j in range(i + 1, num_records):
            p1, h1 = dhash_list[i]
            p2, h2 = dhash_list[j]
            dist = dhash_distance(h1, h2)
            if dist <= 2:
                near_duplicates.append({"file1": p1, "file2": p2, "distance": dist})

    summary = {
        "total_images": len(records),
        "by_category": {
            cat: len([r for r in records if r["category"] == cat]) for cat in categories
        },
        "corrupted_count": len([r for r in records if r["corrupted"]]),
        "exact_duplicate_groups": len(exact_duplicates),
        "exact_duplicate_files_count": sum(len(v) for v in exact_duplicates.values()),
        "near_duplicate_pairs": len(near_duplicates),
        "blurry_count_th50": len([r for r in records if not r["corrupted"] and r["laplacian_var"] < 50]),
        "blurry_count_th30": len([r for r in records if not r["corrupted"] and r["laplacian_var"] < 30]),
        "low_res_count_th480": len([r for r in records if not r["corrupted"] and (r["width"] < 480 or r["height"] < 480)]),
        "extreme_dark_count": len([r for r in records if not r["corrupted"] and r["mean_brightness"] < 30]),
        "extreme_bright_count": len([r for r in records if not r["corrupted"] and r["mean_brightness"] > 225]),
    }
    
    # Save detailed inspection data
    output_json = RESULTS_DIR / "inspection_data.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": summary,
            "exact_duplicates": exact_duplicates,
            "near_duplicates": near_duplicates,
            "records": records
        }, f, indent=2)
        
    print(f"Inspection complete. Json exported to {output_json}")
    print(json.dumps(summary, indent=2))
    return summary, records, exact_duplicates, near_duplicates

if __name__ == "__main__":
    scan_dataset()
