import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

MODEL_PATH = Path(r"e:\nitin\model2\models\road_cleanliness_classifier.pth")
RESULTS_DIR = Path(r"e:\nitin\model2\results\predictions")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['clean_roads', 'slightly_dirty', 'very_dirty']
COLOR_MAP = {
    'clean_roads': (34, 139, 34),       # Green
    'slightly_dirty': (255, 165, 0),    # Orange/Yellow
    'very_dirty': (220, 20, 60)         # Red
}

def load_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 3)
    )
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return model, transform, device

def predict_image(image_path, model, transform, device):
    img_pil = Image.open(image_path).convert('RGB')
    tensor_img = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor_img)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        
    pred_idx = np.argmax(probs)
    pred_class = CLASS_NAMES[pred_idx]
    
    # Calculate Street AIQ Cleanliness Index (0-100%)
    # 100 * p_clean + 50 * p_slightly + 0 * p_very
    cleanliness_score = float(probs[0] * 100.0 + probs[1] * 50.0 + probs[2] * 0.0)
    
    return {
        "predicted_class": pred_class,
        "confidence": float(probs[pred_idx]),
        "cleanliness_index": round(cleanliness_score, 1),
        "probabilities": {
            "clean_roads": float(probs[0]),
            "slightly_dirty": float(probs[1]),
            "very_dirty": float(probs[2])
        }
    }

def visualize_prediction(image_path, result, output_path):
    img = cv2.imread(str(image_path))
    h, w, _ = img.shape
    
    pred_class = result["predicted_class"]
    score = result["cleanliness_index"]
    color = COLOR_MAP[pred_class]
    bgr_color = (color[2], color[1], color[0])
    
    # Draw top banner
    banner_h = int(h * 0.15)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Status text
    status_str = f"Street AIQ: {pred_class.upper().replace('_', ' ')}"
    score_str = f"Cleanliness Score: {score}/100"
    
    cv2.putText(img, status_str, (20, int(banner_h * 0.45)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, bgr_color, 3, cv2.LINE_AA)
    cv2.putText(img, score_str, (20, int(banner_h * 0.85)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
    # Confidence bar at bottom
    bar_h = int(h * 0.05)
    bar_w = int(w * (score / 100.0))
    cv2.rectangle(img, (0, h - bar_h), (bar_w, h), bgr_color, -1)
    
    cv2.imwrite(str(output_path), img)

def batch_predict(image_dir, max_samples=10):
    model, transform, device = load_classifier()
    image_paths = sorted(list(Path(image_dir).rglob("*.*")))
    image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png']][:max_samples]
    
    print(f"\n==================================================")
    print(f"STREET AIQ AUTOMATED ROAD CLEANLINESS PREDICTION")
    print(f"==================================================")
    
    results = []
    for idx, path in enumerate(image_paths, 1):
        res = predict_image(path, model, transform, device)
        results.append((path.name, res))
        
        out_name = f"pred_{idx:02d}_{res['predicted_class']}_{path.name}"
        visualize_prediction(path, res, RESULTS_DIR / out_name)
        
        print(f"[{idx:02d}/{len(image_paths)}] File: {path.name:30s} | Category: {res['predicted_class']:15s} | Cleanliness Score: {res['cleanliness_index']}%")
        
    print(f"\nPredictions complete. Visualizations saved to: {RESULTS_DIR}")
    return results

if __name__ == "__main__":
    test_dir = Path(r"e:\nitin\model2\dataset\classification\val\slightly_dirty")
    batch_predict(test_dir, max_samples=10)
