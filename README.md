# Street AIQ — Real-Time Urban Road Cleanliness & Trash Detection

**Street AIQ** is an end-to-end AI system that predicts hyperlocal road cleanliness ratings (0–100%) and detects road-surface litter in real time using deep neural networks and Generative Adversarial Network (GAN) dataset augmentation.

---

## 🌟 Key Features

* **High-Accuracy Deep Architecture (98.11% Val Accuracy):** Powered by PyTorch Transfer Learning with ResNet50 and Test-Time Augmentation (TTA).
* **GAN Dataset Augmentation:** Trained with a Conditional DCGAN generator synthesizing diverse road textures, lighting conditions, and litter patterns.
* **Real-Time Web Dashboard:** Built with FastAPI + Uvicorn featuring live image drag-and-drop, 1-click test samples, circular score gauges, and municipal action alerts.
* **Auto-Generated Bounding Box Pre-Annotations:** Computer vision saliency pipeline generating over 20,000 candidate trash bounding boxes in standard YOLO format.

---

## 📁 Repository Structure

```
├── app.py                      # Real-time FastAPI web dashboard & inference engine
├── models/
│   └── road_cleanliness_classifier.pth  # Trained 98.11% accuracy PyTorch model
├── road cleanliness/           # Dataset images (clean roads, slightly dirty, very dirty)
├── dataset/
│   └── auto_labeled/           # Auto-generated YOLO bounding box labels
├── results/
│   ├── dataset_inventory.md    # Raw dataset audit & resolution statistics
│   ├── DATASET_REVIEW.md       # Data quality audit report
│   └── resnet50_gan_training_performance.png # Training loss & accuracy curves
├── docs/
│   └── ANNOTATION_GUIDE.md     # 11-scenario human annotation guidelines
└── scripts/
    ├── train_high_accuracy_classifier.py  # ResNet50 + GAN training script
    ├── gan_dataset_augmentor.py# Conditional DCGAN dataset augmentor
    ├── predict_cleanliness.py  # Batch & single-image prediction script
    └── validate_dataset.py     # Annotation schema validation tool
```

---

## 🚀 Quick Start & Usage

### 1. Launch the Real-Time Web Dashboard
```bash
python app.py
```
Open your browser and navigate to: `http://127.0.0.1:8050`

### 2. Run Command-Line Inference
```bash
python scripts/predict_cleanliness.py
```

### 3. Generate GAN Dataset Augmentations
```bash
python scripts/gan_dataset_augmentor.py
```

### 4. Train the Model
```bash
python scripts/train_high_accuracy_classifier.py
```

---

## 📊 Performance Metrics

* **Validation Accuracy:** 98.11%
* **Classes:** `clean_roads` (0), `slightly_dirty` (1), `very_dirty` (2)
* **Score Index:** 0% (Heavily Littered) to 100% (Spotless Clean)
