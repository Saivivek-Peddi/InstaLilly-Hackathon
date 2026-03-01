#!/usr/bin/env python3
"""
Download Data for Pedestrian Detection Model

Datasets:
1. KITTI Object Detection - Primary dataset (requires manual download)
2. CityPersons - Secondary dataset (requires Cityscapes registration)
3. Penn-Fudan - Small test dataset (auto-download)

Usage:
    # Download Penn-Fudan for quick testing
    python models/pedestrian_detector/download_samples.py --test

    # Show instructions for KITTI/CityPersons
    python models/pedestrian_detector/download_samples.py --info
"""

import os
import subprocess
import argparse
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Paths
MODEL_DIR = Path(__file__).parent
DATA_DIR = MODEL_DIR / "data"


def download_file(url: str, dest: Path, desc: str = "Downloading"):
    """Download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"⏭️  {dest.name} already exists, skipping")
        return True

    print(f"📥 {desc}: {dest.name}")

    try:
        with urllib.request.urlopen(url) as response:
            file_size = int(response.headers.get('Content-Length', 0))

            with open(dest, 'wb') as f:
                with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"✅ Downloaded {dest.name}")
        return True

    except Exception as e:
        print(f"❌ Failed to download: {e}")
        if dest.exists():
            dest.unlink()
        return False


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract a zip file."""
    print(f"📦 Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print(f"✅ Extracted to {extract_to}")


def download_penn_fudan():
    """
    Download Penn-Fudan Pedestrian Database.
    Small dataset (~170 images) - good for testing pipeline.
    """
    print("\n" + "="*60)
    print("🚶 PENN-FUDAN PEDESTRIAN DATABASE")
    print("="*60)

    output_dir = DATA_DIR / "penn_fudan"
    zip_path = DATA_DIR / "PennFudanPed.zip"

    url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"

    if output_dir.exists() and any(output_dir.rglob("*.png")):
        print(f"⏭️  Penn-Fudan already downloaded at {output_dir}")
        return True

    if download_file(url, zip_path, "Penn-Fudan dataset"):
        extract_zip(zip_path, DATA_DIR)
        # Rename extracted folder
        extracted_dir = DATA_DIR / "PennFudanPed"
        if extracted_dir.exists():
            if output_dir.exists():
                import shutil
                shutil.rmtree(output_dir)
            extracted_dir.rename(output_dir)
        return True

    return False


def print_kitti_instructions():
    """Print instructions for downloading KITTI dataset."""
    print("\n" + "="*60)
    print("🚗 KITTI OBJECT DETECTION DATASET")
    print("="*60)

    kitti_dir = DATA_DIR / "kitti"

    print("""
KITTI is the primary dataset for autonomous driving pedestrian detection.

📥 Download Instructions:

1. Go to: http://www.cvlibs.net/datasets/kitti/eval_object.php

2. Download these files:
   - "Download left color images of object data set (12 GB)"
     → data_object_image_2.zip
   - "Download training labels of object data set (5 MB)"
     → data_object_label_2.zip

3. Extract to the following structure:
""")
    print(f"   {kitti_dir}/")
    print(f"   └── training/")
    print(f"       ├── image_2/     ← Extract data_object_image_2.zip here")
    print(f"       │   ├── 000000.png")
    print(f"       │   ├── 000001.png")
    print(f"       │   └── ...")
    print(f"       └── label_2/     ← Extract data_object_label_2.zip here")
    print(f"           ├── 000000.txt")
    print(f"           ├── 000001.txt")
    print(f"           └── ...")

    print("""
4. Quick download commands (if you have wget):
""")
    print(f"   mkdir -p {kitti_dir}/training")
    print(f"   cd {kitti_dir}/training")
    print("   # Download requires accepting terms on website first")

    print("""
📊 Dataset Info:
   - ~7,500 training images
   - Labels include: Car, Pedestrian, Cyclist, etc.
   - Real dashcam footage from Karlsruhe, Germany
""")


def print_citypersons_instructions():
    """Print instructions for downloading CityPersons dataset."""
    print("\n" + "="*60)
    print("🏙️  CITYPERSONS DATASET")
    print("="*60)

    city_dir = DATA_DIR / "citypersons"

    print("""
CityPersons is a pedestrian detection benchmark built on Cityscapes.

📥 Download Instructions:

1. Register at: https://www.cityscapes-dataset.com/register/

2. After approval, download:
   - leftImg8bit_trainvaltest.zip (11 GB) - Images
   - gtBboxCityPersons.zip (2 MB) - Pedestrian annotations

3. Extract to the following structure:
""")
    print(f"   {city_dir}/")
    print(f"   ├── leftImg8bit/")
    print(f"   │   ├── train/")
    print(f"   │   │   ├── aachen/")
    print(f"   │   │   │   └── *.png")
    print(f"   │   │   └── .../")
    print(f"   │   └── val/")
    print(f"   └── gtBboxCityPersons/")
    print(f"       ├── train/")
    print(f"       │   ├── aachen/")
    print(f"       │   │   └── *.json")
    print(f"       │   └── .../")
    print(f"       └── val/")

    print("""
📊 Dataset Info:
   - ~5,000 images with pedestrian annotations
   - High-resolution urban scenes (2048x1024)
   - Covers 50 different cities
""")


def verify_datasets():
    """Check which datasets are available."""
    print("\n" + "="*60)
    print("📊 DATASET STATUS")
    print("="*60)

    datasets = {
        "kitti": DATA_DIR / "kitti" / "training" / "image_2",
        "citypersons": DATA_DIR / "citypersons" / "leftImg8bit",
        "penn_fudan": DATA_DIR / "penn_fudan" / "PNGImages",
    }

    total_images = 0

    for name, path in datasets.items():
        if path.exists():
            images = list(path.rglob("*.png")) + list(path.rglob("*.jpg"))
            count = len(images)
            total_images += count
            print(f"  ✅ {name}: {count} images")
        else:
            print(f"  ❌ {name}: not found")

    print(f"\n📊 Total images available: {total_images}")

    if total_images > 0:
        print("\n✅ Ready to prepare data:")
        print("   python models/pedestrian_detector/prepare_data.py")
    else:
        print("\n⚠️  No datasets found. Download at least one dataset.")

    return total_images > 0


def main():
    parser = argparse.ArgumentParser(description="Download pedestrian detection datasets")
    parser.add_argument("--test", action="store_true",
                       help="Download Penn-Fudan for quick testing")
    parser.add_argument("--info", action="store_true",
                       help="Show download instructions for all datasets")
    parser.add_argument("--kitti", action="store_true",
                       help="Show KITTI download instructions")
    parser.add_argument("--citypersons", action="store_true",
                       help="Show CityPersons download instructions")
    args = parser.parse_args()

    print("🚗 Pedestrian Detection - Dataset Download")
    print("="*60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.info or args.kitti:
        print_kitti_instructions()

    if args.info or args.citypersons:
        print_citypersons_instructions()

    if args.test or (not args.info and not args.kitti and not args.citypersons):
        # Default: download Penn-Fudan for testing
        download_penn_fudan()

    verify_datasets()

    print("\n" + "="*60)
    print("📋 NEXT STEPS")
    print("="*60)
    print("""
1. For quick testing (Penn-Fudan only):
   python models/pedestrian_detector/prepare_data.py --test

2. For full training, download KITTI:
   python models/pedestrian_detector/download_samples.py --kitti
   # Follow the instructions, then:
   python models/pedestrian_detector/prepare_data.py

3. Start training:
   python models/pedestrian_detector/train.py --baseline
""")


if __name__ == "__main__":
    main()
