#!/usr/bin/env python3
"""
Extract frames from Kaggle pedestrian video dataset.

The Kaggle pedestrian-dataset contains:
- crosswalk.avi + crosswalk.csv (bounding boxes per frame)
- fourway.avi + fourway.csv
- night.avi + night.csv

This script extracts frames and creates labeled images.
"""

import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

MODEL_DIR = Path(__file__).parent
DATA_DIR = MODEL_DIR / "data"
KAGGLE_DIR = DATA_DIR / "pedestrian_kaggle"
OUTPUT_DIR = DATA_DIR / "kaggle_frames"


def extract_frames_from_video(video_path: Path, csv_path: Path, output_dir: Path, sample_rate: int = 5):
    """
    Extract frames from video with pedestrian annotations.

    Args:
        video_path: Path to .avi video
        csv_path: Path to .csv with bounding boxes
        output_dir: Where to save extracted frames
        sample_rate: Extract every Nth frame to avoid redundancy
    """
    if not video_path.exists():
        print(f"⚠️  Video not found: {video_path}")
        return []

    video_name = video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        has_annotations = True
    else:
        has_annotations = False

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Processing {video_name}: {total_frames} frames")

    records = []
    frame_idx = 0
    saved_count = 0

    with tqdm(total=total_frames // sample_rate) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample every Nth frame
            if frame_idx % sample_rate == 0:
                # Determine if this frame has pedestrians
                has_pedestrian = False
                bbox = None

                if has_annotations and frame_idx < len(df):
                    row = df.iloc[frame_idx]
                    # Check if bounding box is valid (non-zero)
                    if row['w'] > 0 and row['h'] > 0:
                        has_pedestrian = True
                        bbox = [row['x'], row['y'], row['w'], row['h']]

                # Determine label
                if has_pedestrian:
                    label = "pedestrian"
                else:
                    label = "no_pedestrian"

                # Save frame
                filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), frame)

                records.append({
                    "source": f"kaggle_{video_name}",
                    "original_path": str(filepath),
                    "label": label,
                    "pedestrian_count": 1 if has_pedestrian else 0,
                    "bboxes": str([bbox]) if bbox else "[]"
                })

                saved_count += 1
                pbar.update(1)

            frame_idx += 1

    cap.release()
    print(f"  ✅ Saved {saved_count} frames from {video_name}")

    return records


def main():
    print("="*60)
    print("🎬 Extracting frames from Kaggle pedestrian videos")
    print("="*60)

    if not KAGGLE_DIR.exists():
        print(f"❌ Kaggle dataset not found at {KAGGLE_DIR}")
        print("   Run: python download_samples.py --kaggle")
        return

    all_records = []

    # Process each video
    videos = [
        ("crosswalk.avi", "crosswalk.csv"),
        ("fourway.avi", "fourway.csv"),
        ("night.avi", "night.csv"),
    ]

    for video_file, csv_file in videos:
        video_path = KAGGLE_DIR / video_file
        csv_path = KAGGLE_DIR / csv_file

        if video_path.exists():
            records = extract_frames_from_video(
                video_path,
                csv_path,
                OUTPUT_DIR,
                sample_rate=10  # Extract every 10th frame
            )
            all_records.extend(records)

    # Save records summary
    if all_records:
        import json
        with open(OUTPUT_DIR / "frame_records.json", "w") as f:
            json.dump(all_records, f, indent=2)

        print(f"\n✅ Extracted {len(all_records)} frames total")
        print(f"   Saved to: {OUTPUT_DIR}")

        # Count labels
        ped_count = sum(1 for r in all_records if r["label"] == "pedestrian")
        no_ped_count = sum(1 for r in all_records if r["label"] == "no_pedestrian")
        print(f"   Pedestrian frames: {ped_count}")
        print(f"   No pedestrian frames: {no_ped_count}")
    else:
        print("❌ No frames extracted")


if __name__ == "__main__":
    main()
