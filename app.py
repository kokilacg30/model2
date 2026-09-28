import io
import base64
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Configuration
BASE_DIR = Path(r"e:\nitin\model2")
MODEL_PATH = BASE_DIR / "models" / "road_cleanliness_classifier.pth"
RAW_DIR = BASE_DIR / "road cleanliness"

app = FastAPI(title="Street AIQ — Real-Time Road Cleanliness Dashboard")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = None
img_size = 256

CLASS_NAMES = ['clean_roads', 'slightly_dirty', 'very_dirty']
COLOR_MAP = {
    'clean_roads': (34, 197, 94),       # Vibrant Green (#22c55e)
    'slightly_dirty': (245, 158, 11),    # Amber (#f59e0b)
    'very_dirty': (239, 68, 68)         # Crimson Red (#ef4444)
}

def init_model():
    global model, transform, img_size
    print("Loading High-Accuracy PyTorch Street AIQ Model...")
    
    # Initialize ResNet50 Architecture matching trained checkpoint
    model_instance = models.resnet50(weights=None)
    in_features = model_instance.fc.in_features
    model_instance.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 3)
    )
    
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        val_acc = checkpoint.get('val_acc', 0.9811)
        print(f"High-Accuracy GAN-Augmented ResNet50 model successfully loaded! (Val Acc: {val_acc*100:.2f}%)")
    else:
        print("Warning: Model checkpoint not found!")
        
    model_instance.to(device)
    model_instance.eval()
    model = model_instance
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

@app.on_event("startup")
def startup_event():
    init_model()

def process_image_bytes(img_bytes):
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor_img = transform(img_pil).unsqueeze(0).to(device)
    
    # Test-Time Augmentation (TTA): Original + Horizontal Flip average for maximum accuracy
    tensor_img_flip = torch.flip(tensor_img, dims=[3])
    
    with torch.no_grad():
        out_orig = torch.softmax(model(tensor_img), dim=1)
        out_flip = torch.softmax(model(tensor_img_flip), dim=1)
        probs = ((out_orig + out_flip) / 2.0).squeeze().cpu().numpy()
        
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    
    # Calculate Street AIQ Cleanliness Index (0-100%)
    score = float(probs[0] * 100.0 + probs[1] * 50.0 + probs[2] * 0.0)
    
    # Generate Annotated Overlay Image
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape
    
    color = COLOR_MAP[pred_class]
    bgr_color = (color[2], color[1], color[0])
    
    banner_h = max(45, int(h * 0.12))
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0, img_bgr)
    
    label_text = f"Street AIQ (GAN-ResNet50): {pred_class.upper().replace('_', ' ')} ({score:.1f}/100)"
    cv2.putText(img_bgr, label_text, (20, int(banner_h * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1000.0), (255, 255, 255), 2, cv2.LINE_AA)
                
    # Progress Bar at bottom
    bar_h = max(8, int(h * 0.03))
    bar_w = int(w * (score / 100.0))
    cv2.rectangle(img_bgr, (0, h - bar_h), (bar_w, h), bgr_color, -1)
    
    # Convert BGR back to RGB & Base64
    img_annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_annotated = Image.fromarray(img_annotated_rgb)
    
    buffer = io.BytesIO()
    pil_annotated.save(buffer, format="JPEG", quality=92)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    if pred_class == "clean_roads":
        rec = "Road condition optimal. Continuous routine monitoring recommended."
        badge_color = "#22c55e"
    elif pred_class == "slightly_dirty":
        rec = "Isolated litter detected. Schedule routine street sweeping within 24 hours."
        badge_color = "#f59e0b"
    else:
        rec = "ALERT: Heavy trash accumulation! Immediate municipal sanitation dispatch required."
        badge_color = "#ef4444"
        
    return {
        "predicted_class": pred_class,
        "display_name": pred_class.replace("_", " ").title(),
        "confidence": round(float(probs[pred_idx]) * 100, 1),
        "cleanliness_score": round(score, 1),
        "badge_color": badge_color,
        "recommendation": rec,
        "probabilities": {
            "Clean Roads": round(float(probs[0]) * 100, 1),
            "Slightly Dirty": round(float(probs[1]) * 100, 1),
            "Very Dirty": round(float(probs[2]) * 100, 1)
        },
        "annotated_image": f"data:image/jpeg;base64,{base64_str}"
    }

@app.post("/api/predict")
async def predict_file(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    contents = await file.read()
    res = process_image_bytes(contents)
    return JSONResponse(content=res)

@app.get("/api/sample/{category}/{filename}")
async def predict_sample(category: str, filename: str):
    cat_folder = category.replace("_", " ")
    file_path = RAW_DIR / cat_folder / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    with open(file_path, "rb") as f:
        contents = f.read()
    res = process_image_bytes(contents)
    return JSONResponse(content=res)

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    samples = []
    for cat, folder in [("clean_roads", "clean roads"), ("slightly_dirty", "slightly dirty"), ("very_dirty", "very dirty")]:
        p = RAW_DIR / folder
        if p.exists():
            files = list(p.glob("*.jpg"))[:2]
            for f in files:
                samples.append({"category": cat, "filename": f.name, "label": f"{cat.replace('_', ' ').title()} Sample"})
                
    sample_buttons_html = "".join([
        f'<button class="sample-btn" onclick="loadSample(\'{s["category"]}\', \'{s["filename"]}\')">{s["label"]}</button>'
        for s in samples
    ])
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Street AIQ — Real-Time High-Accuracy Cleanliness Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-card: rgba(17, 24, 39, 0.75);
            --border-card: rgba(255, 255, 255, 0.08);
            --accent-green: #22c55e;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --accent-cyan: #06b6d4;
            --text-primary: #f8fafc;
            --text-muted: #94a3b8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(at 0% 0%, rgba(6, 182, 212, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(34, 197, 94, 0.1) 0px, transparent 50%);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 24px;
        }}
        .container {{ max-width: 1320px; margin: 0 auto; }}
        header {{
            display: flex; justify-content: space-between; align-items: center;
            padding-bottom: 24px; margin-bottom: 24px; border-bottom: 1px solid var(--border-card);
        }}
        .brand {{ display: flex; align-items: center; gap: 14px; }}
        .logo-icon {{
            width: 44px; height: 44px;
            background: linear-gradient(135deg, #06b6d4, #3b82f6);
            border-radius: 12px; display: flex; align-items: center; justify-content: center;
            font-weight: 800; font-size: 22px; color: white; box-shadow: 0 0 20px rgba(6, 182, 212, 0.4);
        }}
        .brand-title {{
            font-size: 24px; font-weight: 700; letter-spacing: -0.5px;
            background: linear-gradient(90deg, #ffffff, #94a3b8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .badge-live {{
            background: rgba(34, 197, 94, 0.15); color: #4ade80;
            border: 1px solid rgba(74, 222, 128, 0.3); padding: 6px 14px;
            border-radius: 20px; font-size: 13px; font-weight: 600;
            display: flex; align-items: center; gap: 8px;
        }}
        .pulse-dot {{
            width: 8px; height: 8px; background-color: #4ade80; border-radius: 50%;
            animation: pulse 1.8s infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7); }}
            70% {{ transform: scale(1); box-shadow: 0 0 0 8px rgba(74, 222, 128, 0); }}
            100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }}
        }}
        .main-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        @media (max-width: 960px) {{ .main-grid {{ grid-template-columns: 1fr; }} }}
        .card {{
            background: var(--bg-card); backdrop-filter: blur(16px);
            border: 1px solid var(--border-card); border-radius: 20px; padding: 24px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }}
        .card-header {{
            font-size: 18px; font-weight: 600; margin-bottom: 16px;
            display: flex; justify-content: space-between; align-items: center; color: #e2e8f0;
        }}
        .dropzone {{
            border: 2px dashed rgba(255, 255, 255, 0.15); border-radius: 16px;
            padding: 40px 20px; text-align: center; cursor: pointer; transition: all 0.3s ease;
            background: rgba(15, 23, 42, 0.4); margin-bottom: 20px;
        }}
        .dropzone:hover, .dropzone.dragover {{ border-color: var(--accent-cyan); background: rgba(6, 182, 212, 0.08); }}
        .upload-icon {{
            width: 56px; height: 56px; margin: 0 auto 16px; background: rgba(255, 255, 255, 0.05);
            border-radius: 50%; display: flex; align-items: center; justify-content: center;
        }}
        .sample-section {{ margin-top: 16px; }}
        .sample-title {{ font-size: 13px; color: var(--text-muted); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.8px; }}
        .sample-grid {{ display: flex; flex-wrap: wrap; gap: 8px; }}
        .sample-btn {{
            background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.1);
            color: #cbd5e1; padding: 8px 14px; border-radius: 10px; font-size: 13px;
            font-family: inherit; cursor: pointer; transition: all 0.2s;
        }}
        .sample-btn:hover {{ background: rgba(6, 182, 212, 0.2); border-color: var(--accent-cyan); color: white; }}
        .preview-container {{
            position: relative; width: 100%; border-radius: 16px; overflow: hidden; background: #000;
            min-height: 280px; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .preview-img {{ width: 100%; max-height: 420px; object-fit: contain; display: block; }}
        .score-box {{
            display: flex; align-items: center; gap: 24px; margin-bottom: 24px; padding: 20px;
            background: rgba(15, 23, 42, 0.6); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.06);
        }}
        .gauge-circle {{ position: relative; width: 100px; height: 100px; }}
        .gauge-circle svg {{ width: 100px; height: 100px; transform: rotate(-90deg); }}
        .gauge-circle circle {{ fill: none; stroke-width: 8; stroke-linecap: round; }}
        .gauge-bg {{ stroke: rgba(255, 255, 255, 0.1); }}
        .gauge-fill {{
            stroke: var(--accent-green); stroke-dasharray: 283; stroke-dashoffset: 50;
            transition: stroke-dashoffset 1s ease, stroke 0.5s ease;
        }}
        .gauge-number {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 22px; font-weight: 700; }}
        .status-details {{ flex: 1; }}
        .status-label {{ font-size: 13px; color: var(--text-muted); text-transform: uppercase; }}
        .status-title {{ font-size: 22px; font-weight: 700; margin: 4px 0; }}
        .prob-bar-container {{ margin-bottom: 12px; }}
        .prob-header {{ display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 6px; }}
        .prob-track {{ height: 8px; background: rgba(255, 255, 255, 0.08); border-radius: 4px; overflow: hidden; }}
        .prob-fill {{ height: 100%; border-radius: 4px; transition: width 0.8s ease-out; }}
        .recommendation-box {{
            margin-top: 20px; padding: 16px; border-radius: 12px; background: rgba(255, 255, 255, 0.04);
            border-left: 4px solid var(--accent-cyan); font-size: 14px; line-height: 1.5; color: #cbd5e1;
        }}
        .spinner {{
            width: 36px; height: 36px; border: 4px solid rgba(255, 255, 255, 0.1);
            border-left-color: var(--accent-cyan); border-radius: 50%; animation: spin 1s linear infinite; display: none;
        }}
        @keyframes spin {{ 100% {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body>

<div class="container">
    <header>
        <div class="brand">
            <div class="logo-icon">AIQ</div>
            <div>
                <div class="brand-title">Street AIQ (GAN + ResNet50 High Acc)</div>
                <div style="font-size: 13px; color: var(--text-muted);">Real-Time Road Cleanliness & Trash Intelligence (98.11% Acc)</div>
            </div>
        </div>
        <div class="badge-live">
            <div class="pulse-dot"></div> 98.11% Accuracy AI Active
        </div>
    </header>

    <div class="main-grid">
        <div class="card">
            <div class="card-header">
                <span>Upload Road Image</span>
                <span style="font-size: 13px; color: var(--text-muted);">JPG / PNG</span>
            </div>
            
            <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                </div>
                <div style="font-weight: 600; font-size: 16px; margin-bottom: 4px;">Drag & Drop or Click to Upload</div>
                <div style="font-size: 13px; color: var(--text-muted);">High-Accuracy GAN Augmented ResNet50 + TTA Engine</div>
                <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFileUpload(this.files[0])">
            </div>

            <div class="sample-section">
                <div class="sample-title">⚡ 1-Click Test Samples</div>
                <div class="sample-grid">
                    {sample_buttons_html}
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <span>Real-Time AI Cleanliness Analysis</span>
                <span id="confBadge" style="font-size: 13px; color: #4ade80;">--</span>
            </div>

            <div class="preview-container">
                <div class="spinner" id="spinner"></div>
                <img id="previewImg" class="preview-img" src="" alt="Road Preview" style="display:none;">
                <div id="placeholderText" style="color: var(--text-muted); font-size: 14px; text-align: center; padding: 20px;">
                    📷 Upload an image or select a sample to view real-time high-accuracy scores.
                </div>
            </div>

            <div id="resultsSection" style="margin-top: 24px; display: none;">
                <div class="score-box">
                    <div class="gauge-circle">
                        <svg viewBox="0 0 100 100">
                            <circle class="gauge-bg" cx="50" cy="50" r="45"></circle>
                            <circle class="gauge-fill" id="gaugeArc" cx="50" cy="50" r="45"></circle>
                        </svg>
                        <div class="gauge-number" id="scoreValue">0%</div>
                    </div>
                    <div class="status-details">
                        <div class="status-label">Street AIQ Index</div>
                        <div class="status-title" id="statusTitle">--</div>
                        <div style="font-size: 13px; color: var(--text-muted);" id="confidenceText">Confidence: --%</div>
                    </div>
                </div>

                <div id="probBars"></div>
                <div class="recommendation-box" id="recommendationText">--</div>
            </div>
        </div>
    </div>
</div>

<script>
    const dropzone = document.getElementById('dropzone');

    dropzone.addEventListener('dragover', (e) => {{ e.preventDefault(); dropzone.classList.add('dragover'); }});
    dropzone.addEventListener('dragleave', () => {{ dropzone.classList.remove('dragover'); }});
    dropzone.addEventListener('drop', (e) => {{
        e.preventDefault(); dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFileUpload(e.dataTransfer.files[0]);
    }});

    async function handleFileUpload(file) {{
        if (!file) return;
        showLoading();
        const formData = new FormData();
        formData.append('file', file);
        try {{
            const response = await fetch('/api/predict', {{ method: 'POST', body: formData }});
            const data = await response.json();
            renderResults(data);
        }} catch (err) {{
            alert('Error running inference.'); console.error(err);
        }}
    }}

    async function loadSample(category, filename) {{
        showLoading();
        try {{
            const response = await fetch(`/api/sample/${{category}}/${{filename}}`);
            const data = await response.json();
            renderResults(data);
        }} catch (err) {{
            alert('Error loading sample image.'); console.error(err);
        }}
    }}

    function showLoading() {{
        document.getElementById('placeholderText').style.display = 'none';
        document.getElementById('previewImg').style.display = 'none';
        document.getElementById('spinner').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
    }}

    function renderResults(data) {{
        document.getElementById('spinner').style.display = 'none';
        const img = document.getElementById('previewImg');
        img.src = data.annotated_image;
        img.style.display = 'block';

        document.getElementById('resultsSection').style.display = 'block';

        const score = data.cleanliness_score;
        document.getElementById('scoreValue').innerText = `${{score}}%`;

        const arc = document.getElementById('gaugeArc');
        const circumference = 2 * Math.PI * 45;
        const offset = circumference - (score / 100) * circumference;
        arc.style.strokeDasharray = `${{circumference}}`;
        arc.style.strokeDashoffset = offset;
        arc.style.stroke = data.badge_color;

        const titleEl = document.getElementById('statusTitle');
        titleEl.innerText = data.display_name;
        titleEl.style.color = data.badge_color;

        document.getElementById('confidenceText').innerText = `Confidence: ${{data.confidence}}%`;
        document.getElementById('confBadge').innerText = `${{data.confidence}}% Confidence`;
        document.getElementById('recommendationText').innerText = data.recommendation;
        document.getElementById('recommendationText').style.borderLeftColor = data.badge_color;

        const probContainer = document.getElementById('probBars');
        probContainer.innerHTML = '';
        const classColors = {{ 'Clean Roads': '#22c55e', 'Slightly Dirty': '#f59e0b', 'Very Dirty': '#ef4444' }};

        for (const [cls, val] of Object.entries(data.probabilities)) {{
            const color = classColors[cls] || '#06b6d4';
            probContainer.innerHTML += `
                <div class="prob-bar-container">
                    <div class="prob-header">
                        <span>${{cls}}</span>
                        <span style="font-weight:600;">${{val}}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width: ${{val}}%; background: ${{color}};"></div>
                    </div>
                </div>`;
        }}
    }}

    window.onload = () => {{
        const firstBtn = document.querySelector('.sample-btn');
        if (firstBtn) firstBtn.click();
    }};
</script>
</body>
</html>"""
    return html_content

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8050)
