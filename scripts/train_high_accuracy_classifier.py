import os
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt

# Configuration
AUG_TRAIN_DIR = Path(r"e:\nitin\model2\dataset\classification\train_augmented")
VAL_DIR = Path(r"e:\nitin\model2\dataset\classification\val")
MODEL_DIR = Path(r"e:\nitin\model2\models")
RESULTS_DIR = Path(r"e:\nitin\model2\results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 18
LEARNING_RATE = 5e-4
NUM_CLASSES = 3
CLASSES = ['clean_roads', 'slightly_dirty', 'very_dirty']

def build_augmented_data_loaders():
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(AUG_TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
    
    print(f"Augmented Training Set Size: {len(train_dataset)} images")
    print(f"Validation Set Size: {len(val_dataset)} images")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, train_dataset.class_to_idx

def create_model():
    # Use ResNet50 Transfer Learning for high accuracy feature extraction
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, NUM_CLASSES)
    )
    return model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================")
    print(f"STREET AIQ: GAN-AUGMENTED RESNET50 TRAINING")
    print(f"Device: {device}")
    print(f"==================================================")
    
    train_loader, val_loader, class_to_idx = build_augmented_data_loaders()
    
    model = create_model().to(device)
    
    # Label Smoothing Cross Entropy Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)
                
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_path = MODEL_DIR / "road_cleanliness_classifier.pth"
            torch.save({
                'model_type': 'resnet50',
                'img_size': IMG_SIZE,
                'model_state_dict': model.state_dict(),
                'class_to_idx': class_to_idx,
                'val_acc': val_acc,
                'epoch': epoch
            }, save_path)
            print(f"  --> Checkpoint saved to {save_path} (Val Acc: {val_acc*100:.2f}%)")
            
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    
    # Save training performance chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(range(1, NUM_EPOCHS + 1), history["train_loss"], label="Train Loss", color="blue")
    ax1.plot(range(1, NUM_EPOCHS + 1), history["val_loss"], label="Val Loss", color="red")
    ax1.set_title("ResNet50 + GAN Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(range(1, NUM_EPOCHS + 1), [a*100 for a in history["train_acc"]], label="Train Acc", color="blue")
    ax2.plot(range(1, NUM_EPOCHS + 1), [a*100 for a in history["val_acc"]], label="Val Acc", color="green")
    ax2.set_title("Accuracy Curve (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "resnet50_gan_training_performance.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    train()
