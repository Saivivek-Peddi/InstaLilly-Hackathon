#!/usr/bin/env python3
"""
Full In-Vehicle AI Copilot Demo

Demonstrates the complete pipeline:
- Internal Camera → Drowsiness + Distraction Detection
- External Camera → Pedestrian Detection
- Agent Reasoning → Warning Message
- TTS → Audio Output

Usage:
    python demo_full.py
    python demo_full.py --internal driver.jpg --external road.jpg
"""

import argparse
import json
import sys
from pathlib import Path

# Add project to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from agentic_system.router import InVehicleCopilot


def find_sample_images():
    """Find sample images for demo."""
    samples = {
        "internal": [],
        "external": []
    }

    # Look for drowsiness/distraction samples (internal camera)
    drowsiness_dir = PROJECT_DIR / "models/drowsiness_detector/data/processed/images"
    distraction_dir = PROJECT_DIR / "models/distraction_detector/data/processed/images"

    for d in [drowsiness_dir, distraction_dir]:
        if d.exists():
            for img in list(d.glob("*.jpg"))[:3] + list(d.glob("*.png"))[:3]:
                samples["internal"].append(str(img))

    # Look for pedestrian samples (external camera)
    pedestrian_dir = PROJECT_DIR / "models/pedestrian_detector/data/processed/images"
    penn_fudan_dir = PROJECT_DIR / "models/pedestrian_detector/data/penn_fudan/PNGImages"

    for d in [pedestrian_dir, penn_fudan_dir]:
        if d.exists():
            for img in list(d.glob("*.jpg"))[:3] + list(d.glob("*.png"))[:3]:
                samples["external"].append(str(img))

    return samples


def main():
    parser = argparse.ArgumentParser(description="In-Vehicle AI Copilot Demo")
    parser.add_argument("--internal", type=str, help="Internal camera image (driver)")
    parser.add_argument("--external", type=str, help="External camera image (road)")
    parser.add_argument("--json", type=str, help="JSON input file or string")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS output")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  IN-VEHICLE AI COPILOT - FULL DEMO")
    print("=" * 60)

    # Initialize copilot
    print("\nInitializing copilot...")
    copilot = InVehicleCopilot()

    # Determine input
    if args.json:
        # JSON input
        if args.json.startswith("{"):
            input_json = json.loads(args.json)
        else:
            with open(args.json) as f:
                input_json = json.load(f)
    elif args.internal or args.external:
        # Command line args
        input_json = {}
        if args.internal:
            input_json["internal_camera"] = args.internal
        if args.external:
            input_json["external_camera"] = args.external
    else:
        # Find sample images
        print("\nNo images specified. Looking for samples...")
        samples = find_sample_images()

        if not samples["internal"] and not samples["external"]:
            print("No sample images found!")
            print("\nUsage:")
            print("  python demo_full.py --internal driver.jpg --external road.jpg")
            print('  python demo_full.py --json \'{"internal_camera": "a.jpg", "external_camera": "b.jpg"}\'')
            return

        input_json = {}
        if samples["internal"]:
            input_json["internal_camera"] = samples["internal"][0]
            print(f"  Internal: {samples['internal'][0]}")
        if samples["external"]:
            input_json["external_camera"] = samples["external"][0]
            print(f"  External: {samples['external'][0]}")

    # Run demo
    print("\n" + "-" * 60)
    print("Processing...")
    print("-" * 60)

    result = copilot.process(input_json, speak=not args.no_tts)

    # Display results
    print("\n[DETECTIONS]")
    print(f"  Driver State:    {result['detections']['driver_state']}")
    print(f"  Driver Activity: {result['detections']['driver_activity']}")
    print(f"  Pedestrian:      {'YES' if result['detections']['pedestrian_detected'] else 'NO'}")

    print("\n[DECISION]")
    level = result['decision']['alert_level'].upper()
    colors = {
        "NONE": "\033[92m",      # Green
        "LOW": "\033[93m",       # Yellow
        "MEDIUM": "\033[93m",    # Yellow
        "HIGH": "\033[91m",      # Red
        "CRITICAL": "\033[91m",  # Red
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    color = colors.get(level, RESET)

    print(f"  Alert Level: {color}{BOLD}{level}{RESET}")
    print(f"  Message:     {color}{result['decision']['message']}{RESET}")
    print(f"  Reasoning:   {result['decision']['reasoning']}")

    print("\n[TIMING]")
    print(f"  Drowsiness:  {result['detections']['drowsiness_time_ms']:.0f}ms")
    print(f"  Distraction: {result['detections']['distraction_time_ms']:.0f}ms")
    print(f"  Pedestrian:  {result['detections']['pedestrian_time_ms']:.0f}ms")
    print(f"  Total:       {result['total_time_ms']:.0f}ms")

    if result['audio_file']:
        print(f"\n[AUDIO]")
        print(f"  File: {result['audio_file']}")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60 + "\n")

    # Return JSON for programmatic use
    return result


if __name__ == "__main__":
    main()
