#!/usr/bin/env python3
"""
Data Preparation for Pedestrian Detection Model

Processes KITTI and CityPersons datasets for fine-tuning PaliGemma.

Datasets:
- KITTI: http://www.cvlibs.net/datasets/kitti/eval_object.php
- CityPersons: https://www.cityscapes-dataset.com/

Classes:
- no_pedestrian: No person in frame
- pedestrian: One person detected
- multiple_pedestrians: Multiple people detected

Output format:
- images/ folder with processed images
- prompts.csv with image paths, prompts, and labels
- train/val/test splits
"""

import os
import json
import random
from pathlib import Path
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Paths
MODEL_DIR = Path(__file__).parent
DATA_DIR = MODEL_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"

# Class mapping for detection
CLASSES = ["no_pedestrian", "pedestrian", "multiple_pedestrians"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Image size for PaliGemma
TARGET_SIZE = (224, 224)


def process_kitti_dataset():
    """
    Process KITTI Object Detection dataset.

    Expected structure:
    data/kitti/
    ├── training/
    │   ├── image_2/        # Left color images (.png)
    │   └── label_2/        # Labels (.txt)
    └── testing/
        └── image_2/

    Label format (each line):
    type truncated occluded alpha bbox(4) dimensions(3) location(3) rotation_y

    Where type can be: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc, DontCare
    """
    kitti_dir = DATA_DIR / "kitti"

    # Try different possible structures
    possible_paths = [
        kitti_dir / "training",
        kitti_dir / "data_object_image_2" / "training",
        kitti_dir,
    ]

    training_dir = None
    for p in possible_paths:
        if (p / "image_2").exists() or (p / "images").exists():
            training_dir = p
            break

    if training_dir is None:
        print("⚠️  KITTI dataset not found. Expected structure:")
        print(f"   {kitti_dir}/training/image_2/*.png")
        print(f"   {kitti_dir}/training/label_2/*.txt")
        print("\n   Download from: http://www.cvlibs.net/datasets/kitti/eval_object.php")
        return []

    # Find images and labels directories
    images_dir = training_dir / "image_2"
    if not images_dir.exists():
        images_dir = training_dir / "images"

    labels_dir = training_dir / "label_2"
    if not labels_dir.exists():
        labels_dir = training_dir / "labels"

    records = []
    print(f"📁 Processing KITTI dataset from {training_dir}...")

    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"

        pedestrian_count = 0
        bboxes = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        obj_class = parts[0].lower()
                        # Count pedestrians and cyclists (both are relevant for safety)
                        if obj_class in ["pedestrian", "person_sitting", "cyclist"]:
                            pedestrian_count += 1
                            # KITTI bbox format: left, top, right, bottom
                            bbox = [float(parts[4]), float(parts[5]),
                                   float(parts[6]), float(parts[7])]
                            bboxes.append(bbox)

        # Determine label based on pedestrian count
        if pedestrian_count == 0:
            label = "no_pedestrian"
        elif pedestrian_count == 1:
            label = "pedestrian"
        else:
            label = "multiple_pedestrians"

        records.append({
            "source": "kitti",
            "original_path": str(img_path),
            "label": label,
            "pedestrian_count": pedestrian_count,
            "bboxes": json.dumps(bboxes)
        })

    print(f"  Found {len(records)} images from KITTI")

    # Print class distribution
    labels = [r["label"] for r in records]
    for label in CLASSES:
        count = labels.count(label)
        print(f"    - {label}: {count}")

    return records


def process_citypersons_dataset():
    """
    Process CityPersons dataset (part of Cityscapes).

    Expected structure:
    data/citypersons/
    ├── leftImg8bit/
    │   ├── train/
    │   │   ├── aachen/
    │   │   │   └── *.png
    │   │   └── ...
    │   └── val/
    └── gtBboxCityPersons/
        ├── train/
        │   ├── aachen/
        │   │   └── *.json
        │   └── ...
        └── val/

    Annotation JSON format:
    {
        "imgHeight": 1024,
        "imgWidth": 2048,
        "objects": [
            {"label": "pedestrian", "bbox": [x, y, w, h], ...},
            ...
        ]
    }
    """
    city_dir = DATA_DIR / "citypersons"

    if not city_dir.exists():
        print("⚠️  CityPersons dataset not found. Expected structure:")
        print(f"   {city_dir}/leftImg8bit/train/*/")
        print(f"   {city_dir}/gtBboxCityPersons/train/*/")
        print("\n   Download from: https://www.cityscapes-dataset.com/")
        return []

    records = []
    print(f"📁 Processing CityPersons dataset from {city_dir}...")

    # Process both train and val splits
    for split in ["train", "val"]:
        images_base = city_dir / "leftImg8bit" / split
        labels_base = city_dir / "gtBboxCityPersons" / split

        if not images_base.exists():
            continue

        # Iterate through city folders
        for city_folder in images_base.iterdir():
            if not city_folder.is_dir():
                continue

            for img_path in city_folder.glob("*_leftImg8bit.png"):
                # Find corresponding annotation
                ann_name = img_path.name.replace("_leftImg8bit.png", "_gtBboxCityPersons.json")
                ann_path = labels_base / city_folder.name / ann_name

                pedestrian_count = 0
                bboxes = []

                if ann_path.exists():
                    with open(ann_path, "r") as f:
                        ann = json.load(f)
                        if "objects" in ann:
                            for obj in ann["objects"]:
                                obj_label = obj.get("label", "").lower()
                                if obj_label in ["pedestrian", "person", "rider"]:
                                    pedestrian_count += 1
                                    if "bbox" in obj:
                                        bboxes.append(obj["bbox"])

                # Determine label
                if pedestrian_count == 0:
                    label = "no_pedestrian"
                elif pedestrian_count == 1:
                    label = "pedestrian"
                else:
                    label = "multiple_pedestrians"

                records.append({
                    "source": "citypersons",
                    "original_path": str(img_path),
                    "label": label,
                    "pedestrian_count": pedestrian_count,
                    "bboxes": json.dumps(bboxes)
                })

    print(f"  Found {len(records)} images from CityPersons")

    if records:
        labels = [r["label"] for r in records]
        for label in CLASSES:
            count = labels.count(label)
            print(f"    - {label}: {count}")

    return records


def process_penn_fudan_dataset():
    """
    Process Penn-Fudan Pedestrian dataset (small, for testing).

    Expected structure:
    data/penn_fudan/
    ├── PNGImages/
    └── PedMasks/
    """
    pf_dir = DATA_DIR / "penn_fudan"
    if not pf_dir.exists():
        print("⚠️  Penn-Fudan dataset not found (optional, for testing)")
        return []

    records = []
    print("📁 Processing Penn-Fudan dataset (supplementary)...")

    images_dir = pf_dir / "PNGImages"
    if not images_dir.exists():
        images_dir = pf_dir

    for img_path in images_dir.glob("*.png"):
        # Penn-Fudan images always have pedestrians
        records.append({
            "source": "penn_fudan",
            "original_path": str(img_path),
            "label": "pedestrian",
            "pedestrian_count": 1,
            "bboxes": "[]"
        })

    print(f"  Found {len(records)} images from Penn-Fudan")
    return records


def copy_and_resize_images(records: list, max_samples: int = None) -> list:
    """Copy images to processed folder with consistent naming and size."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    if max_samples:
        # Shuffle to get balanced sample across classes
        random.seed(42)
        random.shuffle(records)
        records = records[:max_samples]

    processed_records = []
    print(f"\n📸 Processing {len(records)} images...")

    for idx, record in enumerate(tqdm(records)):
        try:
            # Load image
            img = Image.open(record["original_path"])

            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

            # Save with new name
            new_filename = f"{record['source']}_{record['label']}_{idx:06d}.jpg"
            new_path = IMAGES_DIR / new_filename
            img.save(new_path, "JPEG", quality=95)

            processed_records.append({
                "filename": new_filename,
                "path": str(new_path),
                "label": record["label"],
                "label_idx": CLASS_TO_IDX[record["label"]],
                "source": record["source"],
                "pedestrian_count": record["pedestrian_count"],
                "bboxes": record["bboxes"]
            })

        except Exception as e:
            print(f"  ⚠️  Error processing {record['original_path']}: {e}")
            continue

    return processed_records


def create_splits(records: list, test_size=0.15, val_size=0.15):
    """Create train/val/test splits with stratification."""
    df = pd.DataFrame(records)

    if len(df) < 10:
        # Too few samples for stratified split
        df["split"] = "train"
        return df

    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=42
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val["label"], random_state=42
    )

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    return pd.concat([train, val, test])


def create_prompt_dataset(df: pd.DataFrame):
    """Create dataset with prompts for PaliGemma fine-tuning."""

    # PaliGemma format: simple task prefix + newline
    # Processor auto-adds <image> tokens, don't add manually
    prompt_template = "detect pedestrian\n"

    # Simple, direct responses for classification
    response_templates = {
        "no_pedestrian": "no_pedestrian",
        "pedestrian": "pedestrian",
        "multiple_pedestrians": "multiple_pedestrians"
    }

    prompts = []
    for _, row in df.iterrows():
        prompts.append({
            "filename": row["filename"],
            "path": row["path"],
            "prompt": prompt_template,
            "response": response_templates[row["label"]],
            "label": row["label"],
            "split": row["split"]
        })

    return pd.DataFrame(prompts)


def main(test_mode: bool = False, max_samples: int = None):
    """
    Main data preparation function.

    Args:
        test_mode: If True, only process a few samples for testing
        max_samples: Maximum number of samples to process (for testing)
    """
    print("="*60)
    print("🚶 Pedestrian Detection - Data Preparation")
    print("="*60)
    print("\nSupported datasets:")
    print("  - KITTI: http://www.cvlibs.net/datasets/kitti/eval_object.php")
    print("  - CityPersons: https://www.cityscapes-dataset.com/")
    print("")

    if test_mode:
        print("🧪 TEST MODE: Processing limited samples")
        max_samples = max_samples or 100

    # Collect records from datasets
    all_records = []
    all_records.extend(process_kitti_dataset())
    all_records.extend(process_citypersons_dataset())
    all_records.extend(process_penn_fudan_dataset())  # Supplementary

    if not all_records:
        print("\n❌ No data found. Please download datasets:")
        print("\n   KITTI (recommended):")
        print("   1. Go to http://www.cvlibs.net/datasets/kitti/eval_object.php")
        print("   2. Download 'Download left color images of object data set (12 GB)'")
        print("   3. Download 'Download training labels of object data set (5 MB)'")
        print(f"   4. Extract to: {DATA_DIR}/kitti/training/")
        print("\n   CityPersons:")
        print("   1. Register at https://www.cityscapes-dataset.com/")
        print("   2. Download leftImg8bit_trainvaltest.zip")
        print("   3. Download gtBboxCityPersons.zip")
        print(f"   4. Extract to: {DATA_DIR}/citypersons/")
        print("\n   For quick testing, run:")
        print("   python models/pedestrian_detector/download_samples.py")
        return

    print(f"\n📊 Total records found: {len(all_records)}")

    # Process images
    processed_records = copy_and_resize_images(all_records, max_samples=max_samples)
    print(f"✅ Processed {len(processed_records)} images")

    if not processed_records:
        print("❌ No images were successfully processed")
        return

    # Create splits
    df = create_splits(processed_records)

    # Create prompts for fine-tuning
    prompt_df = create_prompt_dataset(df)

    # Save metadata
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_DIR / "metadata.csv", index=False)
    prompt_df.to_csv(PROCESSED_DIR / "prompts.csv", index=False)

    # Save class info
    class_info = {
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "num_classes": len(CLASSES)
    }
    with open(PROCESSED_DIR / "class_info.json", "w") as f:
        json.dump(class_info, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("📊 DATASET SUMMARY")
    print("="*60)
    print(f"\nTotal images: {len(df)}")
    print(f"\nSplit distribution:")
    print(df["split"].value_counts().to_string())
    print(f"\nClass distribution:")
    print(df["label"].value_counts().to_string())
    print(f"\nSource distribution:")
    print(df["source"].value_counts().to_string())

    print(f"\n✅ Data saved to: {PROCESSED_DIR}")
    print(f"   - metadata.csv: Image metadata and splits")
    print(f"   - prompts.csv: Prompts for fine-tuning")
    print(f"   - class_info.json: Class mapping")
    print(f"   - images/: Processed images")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare pedestrian detection data")
    parser.add_argument("--test", action="store_true", help="Run in test mode with limited samples")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
    args = parser.parse_args()

    main(test_mode=args.test, max_samples=args.max_samples)
