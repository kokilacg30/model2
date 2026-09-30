import os
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(r"e:\nitin\model2")
YAML_PATH = BASE_DIR / "dataset" / "yolo_data" / "dataset.yaml"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_yolo_detector():
    print("==================================================")
    print("TRAINING YOLOv8 ROAD TRASH OBJECT DETECTOR")
    print("==================================================")
    
    # Load pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")
    
    # Train YOLO detector
    results = model.train(
        data=str(YAML_PATH),
        epochs=15,
        imgsz=640,
        batch=16,
        workers=0,
        name="yolo_trash_run",
        project=str(BASE_DIR / "runs"),
        exist_ok=True,
        save=True
    )
    
    # Save trained model weights to models/yolo_trash_detector.pt
    best_weights = BASE_DIR / "runs" / "yolo_trash_run" / "weights" / "best.pt"
    target_weights = MODEL_DIR / "yolo_trash_detector.pt"
    
    if best_weights.exists():
        shutil.copy2(best_weights, target_weights)
        print(f"\n✅ YOLO Trash Detector trained successfully! Model saved to: {target_weights}")
    else:
        print("\nWarning: Could not locate best.pt weights in runs folder.")

if __name__ == "__main__":
    train_yolo_detector()
