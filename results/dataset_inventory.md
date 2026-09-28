# Street AIQ — Dataset Inventory Report

**Project Name:** Street AIQ — Road-Trash Object Detection & Instance Segmentation  
**Report Date:** September 2026  
**Status:** Raw Dataset Inspection Completed (Pre-Annotation Phase)

---

## Executive Summary

A comprehensive automated and visual inspection of the raw image dataset provided for **Street AIQ** has been completed. The dataset consists of **537 raw road images** collected across diverse road conditions, lighting levels, and surface textures. No automated ground-truth generation or pseudo-labeling has been applied.

---

## Key Dataset Statistics

| Metric | Count | Percentage / Notes |
| :--- | :---: | :--- |
| **Total Raw Images** | **537** | 100.0% |
| **Corrupted / Unreadable Images** | **0** | 0.0% (100% file integrity verified via PIL & OpenCV) |
| **Exact MD5 Duplicate Groups** | **0** | 0.0% |
| **Perceptual Near-Duplicates** | **1 pair** | 2 images (`20260720_150141.jpg` & `IMG-20260721-WA0047.jpg`) |
| **Extremely Blurry Images (< 50 Var)**| **1** | `very dirty/20260722_133415.jpg` (Laplacian Var = 41.68) |
| **Clean Road Images (Hard Negatives)**| **195** | 36.3% (`clean roads` directory) |
| **Visible Trash Images** | **342** | 63.7% (`slightly dirty` + `very dirty`) |
| **Difficult / Ambiguous Scenes** | **34** | Scenes with heavy foliage shadows, small wrappers, edge-cut items |

---

## 1. Category Breakdown

The dataset is partitioned into three raw source subdirectories:

| Directory | Image Count | Percentage | Primary Scene Characteristics |
| :--- | :---: | :---: | :--- |
| `clean roads` | **195** | 36.3% | Clear asphalt/concrete, no visible litter. Ideal for Hard Negative set. |
| `slightly dirty` | **179** | 33.3% | Isolated bottles, small plastic wrappers, scattered roadside trash. |
| `very dirty` | **163** | 30.4% | Dense garbage piles, multi-item plastic/paper clusters, heavy overflow. |
| **Total** | **537** | **100.0%** | |

---

## 2. File Format Breakdown

All files use standard JPEG compression with valid EXIF metadata and standard RGB color spaces:

* `.jpg`: **523 images** (97.4%)
* `.jpeg`: **14 images** (14 images, 2.6%)

---

## 3. Image Resolution Breakdown

The dataset contains high-definition and ultra-high-definition capture resolutions from mobile camera sensors and vehicular dashcams:

| Resolution (WxH) | Aspect Ratio | Image Count | Percentage |
| :--- | :---: | :---: | :---: |
| **1280 x 720** (720p HD) | 16:9 | **308** | 57.3% |
| **4032 x 3024** (12MP 4:3) | 4:3 | **137** | 25.5% |
| **4080 x 2296** | ~16:9 | **47** | 8.8% |
| **2560 x 1440** (2K QHD) | 16:9 | **11** | 2.0% |
| **4032 x 2268** | 16:9 | **10** | 1.9% |
| **3264 x 1840** | ~16:9 | **9** | 1.7% |
| **1600 x 900** | 16:9 | **7** | 1.3% |
| **Vertical & Non-Standard** | Various | **8** | 1.5% |

> [!NOTE]
> All images meet or exceed the minimum recommended input resolution of 640x640 for modern YOLO segmentation architectures (YOLO11s-Seg / YOLOv8s-Seg).

---

## 4. Anomaly & Flagged Image Report

The following images have been flagged for manual reviewer attention prior to or during the annotation workflow:

### A. Perceptual Near-Duplicates
* **Pair 1:** `dataset/raw/very dirty/20260720_150141.jpg` and `dataset/raw/very dirty/IMG-20260721-WA0047.jpg`  
  * *Reason:* Perceptual Hash distance = 0 (same scene captured within fractions of a second or re-saved under WhatsApp compression).  
  * *Action Required:* Keep both in the same split (Training) or send one to review. **Do NOT split across Train and Test.**

### B. Motion / Focus Blur Flag
* `dataset/raw/very dirty/20260722_133415.jpg` (Laplacian Variance = 41.68)  
  * *Reason:* Slight motion blur along the road margin.  
  * *Action Required:* Annotate only clearly recognizable trash outlines; skip unidentifiable blurred smudges.

### C. Vertical / Portrait Captures
* `dataset/raw/clean roads/3024x4032_sample.jpg` and `720x1280_sample.jpg` (2 images total)  
  * *Reason:* Captured in portrait mode (taller than wide).  
  * *Action Required:* Annotators should ensure bounding boxes/polygons cover road regions; standard letterboxing/square padding during model preprocessing will preserve aspect ratio.

---

## 5. Next Steps

1. Review flagged near-duplicate images and assign sequence IDs.
2. Distribute images to CVAT / Roboflow using the guidelines specified in [ANNOTATION_GUIDE.md](file:///e:/nitin/model2/docs/ANNOTATION_GUIDE.md).
3. Track annotation progress against the dataset inventory.
