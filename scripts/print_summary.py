import json
from collections import Counter
from pathlib import Path

with open(r"e:\nitin\model2\results\inspection_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

records = data["records"]
near_dups = data["near_duplicates"]

print("=== IMAGE RESOLUTIONS ===")
resolutions = Counter([f"{r['width']}x{r['height']}" for r in records])
for res, cnt in resolutions.most_common():
    print(f"  {res}: {cnt} images")

print("\n=== EXTENSIONS ===")
exts = Counter([r['ext'] for r in records])
for ext, cnt in exts.most_common():
    print(f"  {ext}: {cnt} images")

print("\n=== NEAR DUPLICATE PAIRS ===")
for p in near_dups:
    print(f"  {p['file1']} <--> {p['file2']} (distance: {p['distance']})")

print("\n=== BLURRY IMAGES (Laplacian Var < 50) ===")
blurry = [r for r in records if r['laplacian_var'] < 50]
for b in blurry:
    print(f"  {b['rel_path']}: Laplacian Var = {b['laplacian_var']}")

print("\n=== SAMPLE RECORDS ===")
for cat in ['clean roads', 'slightly dirty', 'very dirty']:
    sample = [r for r in records if r['category'] == cat][0]
    print(f"Category: {cat} -> Sample: {sample['filename']} ({sample['width']}x{sample['height']}, {sample['size_kb']} KB)")
