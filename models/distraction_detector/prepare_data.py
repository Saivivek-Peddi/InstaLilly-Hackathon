#!/usr/bin/env python3
"""
Data Preparation for Distraction Detection Model

Processes Revitsone-5classes driver distraction dataset.

Classes (5 total):
- safe_driving: Driver focused on road
- texting_phone: Driver texting on phone
- talking_phone: Driver talking on phone
- other_activities: Other distracting activities
- turning: Driver turning/looking away

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

# Class mapping for Revitsone-5classes dataset
CLASSES = [
    "safe_driving",
    "texting_phone",
    "talking_phone",
    "other_activities",
    "turning"
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Human-readable descriptions
CLASS_DESCRIPTIONS = {
    "safe_driving": "Driver is focused on the road with both hands on the steering wheel",
    "texting_phone": "Driver is texting on their phone",
    "talking_phone": "Driver is talking on the phone",
    "other_activities": "Driver is engaged in other distracting activities",
    "turning": "Driver is turning or looking away from the road"
}

# Image size for model
TARGET_SIZE = (224, 224)


def process_revitsone_dataset():
    """Process Revitsone-5classes driver distraction dataset."""
    # Look for the dataset in distraction_alt folder
    revitsone_dir = DATA_DIR / "distraction_alt" / "Revitsone-5classes"

    # Handle nested structure
    if not revitsone_dir.exists():
        revitsone_dir = DATA_DIR / "distraction_alt" / "Revitsone-5classes" / "Revitsone-5classes"

    if not revitsone_dir.exists():
        print("⚠️  Revitsone dataset not found. Run download script first.")
        return []

    records = []
    print("📁 Processing Revitsone-5classes dataset...")
    print(f"   Found directory: {revitsone_dir}")

    # Map folder names to our classes
    folder_to_class = {
        "safe_driving": "safe_driving",
        "texting_phone": "texting_phone",
        "talking_phone": "talking_phone",
        "other_activities": "other_activities",
        "turning": "turning"
    }

    # Process each class folder
    for folder_name, label in folder_to_class.items():
        class_folder = revitsone_dir / folder_name
        if not class_folder.exists():
            print(f"   ⚠️  Folder not found: {folder_name}")
            continue

        for img_path in class_folder.glob("*.jpg"):
            records.append({
                "source": "revitsone",
                "original_path": str(img_path),
                "label": label
            })

        for img_path in class_folder.glob("*.png"):
            records.append({
                "source": "revitsone",
                "original_path": str(img_path),
                "label": label
            })

    print(f"  Found {len(records)} images from Revitsone")

    # Print per-class counts
    from collections import Counter
    class_counts = Counter([r["label"] for r in records])
    for label, count in sorted(class_counts.items()):
        print(f"    {label}: {count}")

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

    prompt_template = """Analyze this image of a driver and classify their current activity.

Categories:
- safe_driving: Focused on driving with hands on wheel
- texting_phone: Using phone to text
- talking_phone: Talking on phone
- other_activities: Other distracting activities
- turning: Turning or looking away from road

Output the activity type, risk level, and recommended action."""

    response_template = """Based on my analysis of the driver's posture and hand positions:

Activity: {label}
Description: {description}
Risk Level: {risk_level}

Recommended Action: {action}"""

    risk_levels = {
        "safe_driving": "low",
        "texting_phone": "high",
        "talking_phone": "medium",
        "other_activities": "medium",
        "turning": "medium"
    }

    actions = {
        "safe_driving": "Continue safe driving practices.",
        "texting_phone": "Please stop texting. I can read messages for you.",
        "talking_phone": "Consider using hands-free mode for calls.",
        "other_activities": "Please focus on driving.",
        "turning": "Keep eyes on the road ahead."
    }

    prompts = []
    for _, row in df.iterrows():
        prompts.append({
            "filename": row["filename"],
            "path": row["path"],
            "prompt": prompt_template,
            "response": response_template.format(
                label=row["label"],
                description=CLASS_DESCRIPTIONS[row["label"]],
                risk_level=risk_levels[row["label"]],
                action=actions[row["label"]]
            ),
            "label": row["label"],
            "split": row["split"]
        })

    return pd.DataFrame(prompts)


def main():
    print("="*60)
    print("📱 Distraction Detection - Data Preparation")
    print("="*60)

    # Collect records
    all_records = process_revitsone_dataset()

    if not all_records:
        print("\n❌ No data found. Please download the dataset first:")
        print("   kaggle datasets download -d robinreni/revitsone-5class")
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
        "descriptions": CLASS_DESCRIPTIONS,
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

    print(f"\n✅ Data saved to: {PROCESSED_DIR}")
    print(f"   - metadata.csv: Image metadata and splits")
    print(f"   - prompts.csv: Prompts for fine-tuning")
    print(f"   - class_info.json: Class mapping")
    print(f"   - images/: Processed images")


if __name__ == "__main__":
    main()
