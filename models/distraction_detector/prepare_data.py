#!/usr/bin/env python3
"""
Data Preparation for Distraction Detection Model

Processes State Farm Distracted Driver Detection dataset.

Classes (10 total):
- c0: safe driving
- c1: texting - right
- c2: talking on the phone - right
- c3: texting - left
- c4: talking on the phone - left
- c5: operating the radio
- c6: drinking
- c7: reaching behind
- c8: hair and makeup
- c9: talking to passenger

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

# Class mapping for State Farm dataset
CLASSES = [
    "safe_driving",      # c0
    "texting_right",     # c1
    "phone_right",       # c2
    "texting_left",      # c3
    "phone_left",        # c4
    "operating_radio",   # c5
    "drinking",          # c6
    "reaching_behind",   # c7
    "hair_makeup",       # c8
    "talking_passenger"  # c9
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Map State Farm folder names to our classes
SF_TO_CLASS = {
    "c0": "safe_driving",
    "c1": "texting_right",
    "c2": "phone_right",
    "c3": "texting_left",
    "c4": "phone_left",
    "c5": "operating_radio",
    "c6": "drinking",
    "c7": "reaching_behind",
    "c8": "hair_makeup",
    "c9": "talking_passenger"
}

# Human-readable descriptions
CLASS_DESCRIPTIONS = {
    "safe_driving": "Driver is focused on the road with both hands on the steering wheel",
    "texting_right": "Driver is texting using their right hand",
    "phone_right": "Driver is talking on the phone held in their right hand",
    "texting_left": "Driver is texting using their left hand",
    "phone_left": "Driver is talking on the phone held in their left hand",
    "operating_radio": "Driver is adjusting the radio or dashboard controls",
    "drinking": "Driver is drinking a beverage",
    "reaching_behind": "Driver is reaching behind to the back seat",
    "hair_makeup": "Driver is adjusting hair or applying makeup",
    "talking_passenger": "Driver is turned talking to a passenger"
}

# Image size for model
TARGET_SIZE = (224, 224)


def process_state_farm_dataset():
    """Process State Farm Distracted Driver dataset."""
    sf_dir = DATA_DIR / "state_farm"
    if not sf_dir.exists():
        print("⚠️  State Farm dataset not found. Run download script first.")
        return []

    records = []
    print("📁 Processing State Farm dataset...")

    # Look for train folder
    train_dir = None
    for possible_path in [sf_dir / "train", sf_dir / "imgs" / "train", sf_dir]:
        if possible_path.exists():
            # Check if it has c0, c1, etc. folders
            if any((possible_path / f"c{i}").exists() for i in range(10)):
                train_dir = possible_path
                break

    if train_dir is None:
        # Search recursively
        for folder in sf_dir.rglob("c0"):
            if folder.is_dir():
                train_dir = folder.parent
                break

    if train_dir is None:
        print("⚠️  Could not find State Farm train folder structure")
        print(f"   Looking in: {sf_dir}")
        print(f"   Contents: {list(sf_dir.iterdir())[:10]}")
        return []

    print(f"   Found train directory: {train_dir}")

    # Process each class folder
    for class_idx in range(10):
        class_folder = train_dir / f"c{class_idx}"
        if not class_folder.exists():
            continue

        label = SF_TO_CLASS[f"c{class_idx}"]

        for img_path in class_folder.glob("*.jpg"):
            records.append({
                "source": "state_farm",
                "original_path": str(img_path),
                "label": label,
                "original_class": f"c{class_idx}"
            })

        for img_path in class_folder.glob("*.png"):
            records.append({
                "source": "state_farm",
                "original_path": str(img_path),
                "label": label,
                "original_class": f"c{class_idx}"
            })

    print(f"  Found {len(records)} images from State Farm")

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
                "source": record["source"],
                "description": CLASS_DESCRIPTIONS[record["label"]]
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
- texting_right: Using phone to text with right hand
- texting_left: Using phone to text with left hand
- phone_right: Talking on phone held in right hand
- phone_left: Talking on phone held in left hand
- operating_radio: Adjusting radio/dashboard controls
- drinking: Drinking a beverage
- reaching_behind: Reaching to back seat
- hair_makeup: Fixing hair or makeup
- talking_passenger: Turned talking to passenger

Output the activity type, risk level, and recommended action."""

    response_template = """Based on my analysis of the driver's posture and hand positions:

Activity: {label}
Description: {description}
Risk Level: {risk_level}

Recommended Action: {action}"""

    risk_levels = {
        "safe_driving": "low",
        "texting_right": "high",
        "phone_right": "medium",
        "texting_left": "high",
        "phone_left": "medium",
        "operating_radio": "low",
        "drinking": "low",
        "reaching_behind": "high",
        "hair_makeup": "medium",
        "talking_passenger": "medium"
    }

    actions = {
        "safe_driving": "Continue safe driving practices.",
        "texting_right": "⚠️ Please stop texting. I can read messages for you.",
        "phone_right": "Consider using hands-free mode for calls.",
        "texting_left": "⚠️ Please stop texting. I can read messages for you.",
        "phone_left": "Consider using hands-free mode for calls.",
        "operating_radio": "Keep adjustments brief. I can help with controls.",
        "drinking": "Place beverage in holder when not drinking.",
        "reaching_behind": "⚠️ Pull over safely if you need to reach behind.",
        "hair_makeup": "Please focus on driving. Adjust appearance when parked.",
        "talking_passenger": "Keep eyes on the road while conversing."
    }

    prompts = []
    for _, row in df.iterrows():
        prompts.append({
            "filename": row["filename"],
            "path": row["path"],
            "prompt": prompt_template,
            "response": response_template.format(
                label=row["label"],
                description=row["description"],
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
    all_records = process_state_farm_dataset()

    if not all_records:
        print("\n❌ No data found. Please run the download script first:")
        print("   python scripts/download_data.py --model distraction")
        print("\n   Note: You may need to accept competition rules at:")
        print("   https://www.kaggle.com/c/state-farm-distracted-driver-detection")
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
