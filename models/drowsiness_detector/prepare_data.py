#!/usr/bin/env python3
"""
Data Preparation for Drowsiness Detection Model

Processes raw datasets into a unified format for fine-tuning.

Classes:
- alert: Driver is fully awake and attentive
- drowsy: Driver showing signs of fatigue
- yawning: Driver is yawning
- eyes_closed: Driver's eyes are closed

Output format:
- images/ folder with processed images
- metadata.csv with image paths and labels
- train/val/test splits
"""

import os
import json
import shutil
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

# Class mapping
CLASSES = ["alert", "drowsy", "yawning", "eyes_closed"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Image size for model
TARGET_SIZE = (224, 224)


def process_ddd_dataset():
    """Process Driver Drowsiness Dataset (DDD)."""
    ddd_dir = DATA_DIR / "ddd"
    if not ddd_dir.exists():
        print("⚠️  DDD dataset not found. Run download script first.")
        return []

    records = []
    print("📁 Processing DDD dataset...")

    # DDD structure: typically has Drowsy and Non Drowsy folders
    for class_folder in ddd_dir.rglob("*"):
        if class_folder.is_dir():
            folder_name = class_folder.name.lower()

            # Map folder names to our classes
            if "drowsy" in folder_name and "non" not in folder_name:
                label = "drowsy"
            elif "non" in folder_name or "alert" in folder_name or "awake" in folder_name:
                label = "alert"
            elif "yawn" in folder_name:
                label = "yawning"
            elif "close" in folder_name:
                label = "eyes_closed"
            else:
                continue

            for img_path in class_folder.glob("*.jpg"):
                records.append({"source": "ddd", "original_path": str(img_path), "label": label})
            for img_path in class_folder.glob("*.png"):
                records.append({"source": "ddd", "original_path": str(img_path), "label": label})

    print(f"  Found {len(records)} images from DDD")
    return records


def process_mrl_dataset():
    """Process MRL Eye Dataset."""
    mrl_dir = DATA_DIR / "mrl_eyes"
    if not mrl_dir.exists():
        print("⚠️  MRL dataset not found. Run download script first.")
        return []

    records = []
    print("📁 Processing MRL Eye dataset...")

    # MRL has open and closed eyes
    for class_folder in mrl_dir.rglob("*"):
        if class_folder.is_dir():
            folder_name = class_folder.name.lower()

            if "open" in folder_name:
                label = "alert"
            elif "close" in folder_name:
                label = "eyes_closed"
            else:
                continue

            for img_path in class_folder.glob("*.*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    records.append({"source": "mrl", "original_path": str(img_path), "label": label})

    print(f"  Found {len(records)} images from MRL")
    return records


def process_fl3d_dataset():
    """Process FL3D (Frame-Level Driver Drowsiness Detection) dataset."""
    fl3d_dir = DATA_DIR / "fl3d" / "classification_frames"
    if not fl3d_dir.exists():
        print("⚠️  FL3D dataset not found. Run download script first.")
        return []

    records = []
    print("📁 Processing FL3D dataset...")

    # FL3D has participant folders with annotations_final.json
    for participant_dir in fl3d_dir.iterdir():
        if not participant_dir.is_dir():
            continue

        annotations_file = participant_dir / "annotations_final.json"
        if not annotations_file.exists():
            continue

        # Load annotations
        with open(annotations_file) as f:
            annotations = json.load(f)

        for filename, data in annotations.items():
            img_path = participant_dir / filename
            if not img_path.exists():
                continue

            # Map FL3D states to our classes
            driver_state = data.get("driver_state", "").lower()
            if driver_state == "alert":
                label = "alert"
            elif driver_state in ["drowsy", "sleepy"]:
                label = "drowsy"
            elif "yawn" in driver_state:
                label = "yawning"
            else:
                label = "alert"  # Default

            records.append({"source": "fl3d", "original_path": str(img_path), "label": label})

    print(f"  Found {len(records)} images from FL3D")
    return records


def process_drowsiness_extra_dataset():
    """Process drowsiness extra dataset with eyes and yawn classes."""
    extra_dir = DATA_DIR / "drowsiness_extra" / "train"
    if not extra_dir.exists():
        print("⚠️  Drowsiness extra dataset not found.")
        return []

    records = []
    print("📁 Processing Drowsiness Extra dataset...")

    # Map folder names to our classes
    folder_mapping = {
        "Closed": "eyes_closed",
        "Open": "alert",
        "yawn": "yawning",
        "no_yawn": "alert"
    }

    for folder_name, label in folder_mapping.items():
        folder_path = extra_dir / folder_name
        if not folder_path.exists():
            continue

        for img_path in folder_path.glob("*.*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                records.append({"source": "drowsiness_extra", "original_path": str(img_path), "label": label})

    print(f"  Found {len(records)} images from Drowsiness Extra")
    return records


def copy_and_resize_images(records: list) -> list:
    """Copy images to processed folder with consistent naming and size."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    processed_records = []
    print(f"\n📸 Processing {len(records)} images...")

    for idx, record in enumerate(tqdm(records)):
        try:
            # Load and resize image
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
                "source": record["source"]
            })

        except Exception as e:
            print(f"  ⚠️  Error processing {record['original_path']}: {e}")
            continue

    return processed_records


def create_splits(records: list, test_size=0.15, val_size=0.15):
    """Create train/val/test splits."""
    df = pd.DataFrame(records)

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
    """Create dataset with prompts for fine-tuning."""

    prompt_template = """Analyze this image of a driver and determine their alertness state.

Categories:
- alert: Driver is fully awake and attentive
- drowsy: Driver showing signs of fatigue (heavy eyelids, slow movements)
- yawning: Driver is yawning
- eyes_closed: Driver's eyes are closed

Output the driver's state and your confidence level."""

    response_template = """Based on my analysis of the driver's facial features and behavior:

State: {label}
Confidence: high

Indicators observed:
{indicators}"""

    indicator_map = {
        "alert": "- Eyes fully open\n- Attentive gaze\n- Normal posture",
        "drowsy": "- Heavy eyelids\n- Slow blink rate\n- Reduced alertness",
        "yawning": "- Mouth wide open\n- Yawning motion detected",
        "eyes_closed": "- Eyes fully closed\n- Possible microsleep"
    }

    prompts = []
    for _, row in df.iterrows():
        prompts.append({
            "filename": row["filename"],
            "path": row["path"],
            "prompt": prompt_template,
            "response": response_template.format(
                label=row["label"],
                indicators=indicator_map[row["label"]]
            ),
            "label": row["label"],
            "split": row["split"]
        })

    return pd.DataFrame(prompts)


def main():
    print("="*60)
    print("🚗 Drowsiness Detection - Data Preparation")
    print("="*60)

    # Collect records from all datasets
    all_records = []
    all_records.extend(process_ddd_dataset())
    all_records.extend(process_mrl_dataset())
    all_records.extend(process_fl3d_dataset())
    all_records.extend(process_drowsiness_extra_dataset())

    if not all_records:
        print("\n❌ No data found. Please run the download script first:")
        print("   python scripts/download_data.py --model drowsiness")
        return

    print(f"\n📊 Total records found: {len(all_records)}")

    # Process images
    processed_records = copy_and_resize_images(all_records)
    print(f"✅ Processed {len(processed_records)} images")

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
    main()
