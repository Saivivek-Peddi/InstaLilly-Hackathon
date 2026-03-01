#!/usr/bin/env python3
"""
Data Download Script for In-Vehicle AI Copilot

Downloads datasets for:
- Drowsiness Detection (Person B)
- Distraction Detection (Person B)
- Pedestrian Detection (Person A)
- Voice Commands (Person A)

Usage:
    python scripts/download_data.py --model drowsiness
    python scripts/download_data.py --model distraction
    python scripts/download_data.py --model all
"""

import os
import subprocess
import argparse
from pathlib import Path
import zipfile
import shutil

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"


def run_kaggle_download(dataset: str, output_dir: Path):
    """Download a dataset from Kaggle."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {dataset} to {output_dir}")

    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(output_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Try competition download if dataset download fails
        if "competition" in dataset or "state-farm" in dataset.lower():
            cmd = ["kaggle", "competitions", "download", "-c", dataset.split("/")[-1], "-p", str(output_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Failed to download {dataset}")
        print(result.stderr)
        return False

    print(f"✅ Downloaded {dataset}")
    return True


def extract_zip_files(directory: Path):
    """Extract all zip files in a directory."""
    for zip_file in directory.glob("*.zip"):
        print(f"📦 Extracting {zip_file.name}")
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(directory)
        # Optionally remove zip after extraction
        # zip_file.unlink()


def download_drowsiness_data():
    """Download drowsiness detection datasets."""
    print("\n" + "="*60)
    print("🚗 DROWSINESS DETECTION DATASETS")
    print("="*60)

    data_dir = MODELS_DIR / "drowsiness_detector" / "data"

    datasets = [
        # Main drowsiness dataset
        ("ismailnasri20/driver-drowsiness-dataset-ddd", "ddd"),
        # MRL Eye dataset for eye state
        ("prasadvpatil/mrl-dataset", "mrl_eyes"),
        # Frame-level drowsiness
        ("matjazmuc/frame-level-driver-drowsiness-detection-fl3d", "fl3d"),
        # Additional drowsiness dataset
        ("dheerajperumandla/drowsiness-dataset", "drowsiness_extra"),
    ]

    for dataset, folder in datasets:
        output = data_dir / folder
        if output.exists() and any(output.iterdir()):
            print(f"⏭️  Skipping {dataset} (already exists)")
            continue
        run_kaggle_download(dataset, output)
        extract_zip_files(output)

    print(f"\n✅ Drowsiness data saved to: {data_dir}")
    return data_dir


def download_distraction_data():
    """Download distraction detection datasets."""
    print("\n" + "="*60)
    print("📱 DISTRACTION DETECTION DATASETS")
    print("="*60)

    data_dir = MODELS_DIR / "distraction_detector" / "data"

    # State Farm is a competition dataset
    output = data_dir / "state_farm"

    if output.exists() and any(output.iterdir()):
        print(f"⏭️  Skipping State Farm (already exists)")
    else:
        print("📥 Downloading State Farm Distracted Driver Detection...")
        output.mkdir(parents=True, exist_ok=True)

        # Try competition download
        cmd = ["kaggle", "competitions", "download", "-c", "state-farm-distracted-driver-detection", "-p", str(output)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("⚠️  Competition download failed. You may need to:")
            print("   1. Go to https://www.kaggle.com/c/state-farm-distracted-driver-detection")
            print("   2. Accept the competition rules")
            print("   3. Re-run this script")
            print(f"   Error: {result.stderr}")
        else:
            extract_zip_files(output)
            print("✅ State Farm dataset downloaded")

    print(f"\n✅ Distraction data saved to: {data_dir}")
    return data_dir


def download_pedestrian_data():
    """Download pedestrian detection datasets."""
    print("\n" + "="*60)
    print("🚶 PEDESTRIAN DETECTION DATASETS")
    print("="*60)

    data_dir = MODELS_DIR / "pedestrian_detector" / "data"

    # KITTI and CityPersons require manual download
    print("⚠️  Pedestrian datasets require manual download:")
    print("   - KITTI: http://www.cvlibs.net/datasets/kitti/eval_object.php")
    print("   - CityPersons: https://www.cityscapes-dataset.com/")
    print("")
    print("   Alternative Kaggle datasets:")

    datasets = [
        ("smeschke/pedestrian-dataset", "pedestrian_kaggle"),
    ]

    for dataset, folder in datasets:
        output = data_dir / folder
        if output.exists() and any(output.iterdir()):
            print(f"⏭️  Skipping {dataset} (already exists)")
            continue
        run_kaggle_download(dataset, output)
        extract_zip_files(output)

    return data_dir


def download_voice_data():
    """Download voice command datasets."""
    print("\n" + "="*60)
    print("🎤 VOICE COMMAND DATASETS")
    print("="*60)

    data_dir = MODELS_DIR / "voice_assistant" / "data"

    datasets = [
        ("oortdatahub/car-command", "car_command"),
    ]

    for dataset, folder in datasets:
        output = data_dir / folder
        if output.exists() and any(output.iterdir()):
            print(f"⏭️  Skipping {dataset} (already exists)")
            continue
        run_kaggle_download(dataset, output)
        extract_zip_files(output)

    return data_dir


def print_data_summary():
    """Print summary of downloaded data."""
    print("\n" + "="*60)
    print("📊 DATA SUMMARY")
    print("="*60)

    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir():
            data_dir = model_dir / "data"
            if data_dir.exists():
                total_files = sum(1 for _ in data_dir.rglob("*") if _.is_file())
                total_size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
                size_mb = total_size / (1024 * 1024)
                print(f"  {model_dir.name}: {total_files} files, {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for In-Vehicle AI Copilot")
    parser.add_argument(
        "--model",
        choices=["drowsiness", "distraction", "pedestrian", "voice", "all", "person_b"],
        default="person_b",
        help="Which model's data to download"
    )
    args = parser.parse_args()

    print("🚗 In-Vehicle AI Copilot - Data Download")
    print("="*60)

    if args.model in ["drowsiness", "all", "person_b"]:
        download_drowsiness_data()

    if args.model in ["distraction", "all", "person_b"]:
        download_distraction_data()

    if args.model in ["pedestrian", "all"]:
        download_pedestrian_data()

    if args.model in ["voice", "all"]:
        download_voice_data()

    print_data_summary()
    print("\n✅ Data download complete!")


if __name__ == "__main__":
    main()
