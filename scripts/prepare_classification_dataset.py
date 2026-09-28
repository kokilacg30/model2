import os
import random
import shutil
from pathlib import Path

def prepare_split(seed=42, val_ratio=0.2):
    random.seed(seed)
    
    base_dir = Path(r"e:\nitin\model2")
    raw_dir = base_dir / "road cleanliness"
    output_dir = base_dir / "dataset" / "classification"
    
    category_mapping = {
        "clean roads": "clean_roads",
        "slightly dirty": "slightly_dirty",
        "very dirty": "very_dirty"
    }
    
    # Recreate clean output directories
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    for target_class in category_mapping.values():
        (train_dir / target_class).mkdir(parents=True, exist_ok=True)
        (val_dir / target_class).mkdir(parents=True, exist_ok=True)
        
    summary = {}
    
    for raw_folder, class_name in category_mapping.items():
        src_path = raw_dir / raw_folder
        images = list(src_path.glob("*.*"))
        images = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
        
        random.shuffle(images)
        val_count = int(len(images) * val_ratio)
        val_images = images[:val_count]
        train_images = images[val_count:]
        
        for img in train_images:
            shutil.copy2(img, train_dir / class_name / img.name)
            
        for img in val_images:
            shutil.copy2(img, val_dir / class_name / img.name)
            
        summary[class_name] = {
            "total": len(images),
            "train": len(train_images),
            "val": len(val_images)
        }
        
    print("Classification dataset split completed successfully!")
    print("Dataset Summary:", summary)

if __name__ == "__main__":
    prepare_split()
