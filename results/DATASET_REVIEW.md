# Street AIQ — Dataset Data Quality & Action Review

**Project Name:** Street AIQ — Road-Trash Object Detection & Instance Segmentation  
**Document:** Data Quality Audit & Recommended Action Plan  
**Status:** Approved for Annotation Pipeline Setup

---

## 1. Quality Audit Summary

| Category | Inspected Count | Status | Action Required |
| :--- | :---: | :---: | :--- |
| **Corrupted Files** | 537 | **PASSED** (0 corrupted) | None. All files readable by CVAT/Roboflow/PyTorch. |
| **Exact Byte Duplicates (MD5)** | 537 | **PASSED** (0 exact duplicates) | None. |
| **Perceptual Near-Duplicates** | 537 | **FLAGGED** (1 pair found) | Keep pair in same split (`train`). Do not split across train/test. |
| **Low-Resolution Images (<480p)** | 537 | **PASSED** (0 images <480p) | Standard resizing to 640x640 during training. |
| **Motion Blur (<50 LapVar)** | 537 | **FLAGGED** (1 image found) | Manual review during annotation. Annotate visible trash only. |
| **Non-Road / Irrelevant Scenes** | 537 | **PASSED** (0 non-road) | All images contain valid road surface contexts. |
| **Clean Road Hard Negatives** | 195 | **VERIFIED** | Reserve 15-20% of clean roads for Val/Test FP evaluation. |

---

## 2. Detailed Findings & Recommended Actions

### Rule: DO NOT Auto-Delete Any Image
All raw images are retained in `dataset/raw/`. No automated script or process will purge images.

### Action Item 1: Perceptual Near-Duplicate Handling
* **Identified Pair:**
  1. `dataset/raw/very dirty/20260720_150141.jpg`
  2. `dataset/raw/very dirty/IMG-20260721-WA0047.jpg`
* **Risk:** If image 1 is placed in `train` and image 2 is placed in `test`, the evaluation metrics will suffer from **data leakage**, artificially inflating test performance.
* **Recommended Action:**
  * Assign both images to the **Training Split (`train`)**.
  * Keep `test` split strictly isolated with independent geographical/temporal captures.

### Action Item 2: Blurry Image Handling
* **Identified Image:**
  1. `dataset/raw/very dirty/20260722_133415.jpg` (Laplacian Variance: 41.68)
* **Risk:** Annotators might draw guesswork polygons around out-of-focus smudges.
* **Recommended Action:**
  * Annotate ONLY distinct trash items where object borders are visually distinguishable.
  * If a smudge cannot be confidently identified as waste, leave it unannotated.

### Action Item 3: Hard Negative Integration Strategy
* **Count:** 195 Clean Road images (`dataset/raw/clean_roads/`).
* **Role in Training:**
  * Clean road images contain NO annotations (empty label `.txt` files in YOLO format).
  * Including hard negatives during model training penalizes the network for predicting false positives on potholes, shadows, leaves, oil spots, and road markings.
* **Recommended Action:**
  * Include 145 clean road images in `train/` (empty `.txt` labels).
  * Include 25 clean road images in `val/` to monitor False Positive rate during training.
  * Include 25 clean road images in `test/` for final model calibration.
  * Maintain a dedicated mirror folder at `dataset/hard_negatives/` for isolated false-positive evaluation benchmark runs.

---

## 3. Pre-Annotation Action Checklist

- [x] Create project subdirectories (`raw`, `annotation`, `images`, `labels`, `train`, `val`, `test`, `hard_negatives`).
- [x] Run automated image integrity scan (`scripts/dataset_statistics.py`).
- [x] Verify resolution and color space compatibility.
- [ ] Upload dataset batch to CVAT / Roboflow platform.
- [ ] Provide annotators with [docs/ANNOTATION_GUIDE.md](file:///e:/nitin/model2/docs/ANNOTATION_GUIDE.md).
- [ ] Run `scripts/validate_dataset.py` after export to ensure 100% schema compliance.
