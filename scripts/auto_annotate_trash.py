import os
import cv2
import numpy as np
from pathlib import Path

def generate_pseudo_labels(image_path, min_area=300, max_area=50000):
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    h, w, _ = img.shape
    
    # Convert to LAB and HSV color spaces for trash contrast detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate local contrast saliency
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    
    # Multi-threshold binarization for debris
    _, thresh_lap = cv2.threshold(laplacian_abs, 40, 255, cv2.THRESH_BINARY)
    
    # Color anomaly mask (bright whites, vibrant plastics, foil reflections)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    
    plastic_mask = cv2.inRange(hsv, np.array([0, 50, 100]), np.array([180, 255, 255]))
    white_paper_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 30, 255]))
    
    combined_mask = cv2.bitwise_or(thresh_lap, plastic_mask)
    combined_mask = cv2.bitwise_or(combined_mask, white_paper_mask)
    
    # Focus search on lower 75% of image (road surface)
    roi_mask = np.zeros_like(combined_mask)
    roi_mask[int(h * 0.25):, :] = 255
    combined_mask = cv2.bitwise_and(combined_mask, roi_mask)
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # Filter extreme aspect ratios (e.g. long thin lane markings)
            aspect_ratio = bw / float(bh)
            if 0.2 <= aspect_ratio <= 5.0:
                # Normalize YOLO format: class_id x_center y_center width height
                x_center = (x + bw / 2.0) / w
                y_center = (y + bh / 2.0) / h
                norm_w = bw / float(w)
                norm_h = bh / float(h)
                
                yolo_boxes.append((0, round(x_center, 6), round(y_center, 6), round(norm_w, 6), round(norm_h, 6)))
                
    return yolo_boxes

def auto_annotate_dataset():
    base_dir = Path(r"e:\nitin\model2")
    raw_dir = base_dir / "road cleanliness"
    out_dir = base_dir / "dataset" / "auto_labeled"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    categories = ["slightly dirty", "very dirty"]
    processed = 0
    total_labels = 0
    
    for cat in categories:
        cat_path = raw_dir / cat
        for img_path in cat_path.glob("*.*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            boxes = generate_pseudo_labels(img_path)
            
            # Save label file
            lbl_file = out_dir / "labels" / f"{img_path.stem}.txt"
            with open(lbl_file, "w", encoding="utf-8") as f:
                for b in boxes:
                    f.write(f"{b[0]} {b[1]} {b[2]} {b[3]} {b[4]}\n")
                    
            processed += 1
            total_labels += len(boxes)
            
    print(f"Auto-annotation generated {total_labels} candidate trash bounding boxes across {processed} images!")
    print(f"Auto-labeled files stored at: {out_dir}")

if __name__ == "__main__":
    auto_annotate_dataset()
