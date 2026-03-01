#!/usr/bin/env python3
"""
Demo Script for In-Vehicle AI Copilot

Tests the full agentic pipeline:
1. Pedestrian Detection (simulated or via API)
2. Agent Decision (via Ollama)
3. Warning Execution (text alerts)

Usage:
    # Test without Ollama (rule-based)
    python demo.py --no-ollama

    # Test with Ollama agent
    python demo.py

    # Test with detection server
    python demo.py --detection-server http://VM_IP:8000
"""

import argparse
import time
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_system import SafetyPipeline, SafetyAgent, trigger_warning


def demo_simple():
    """Simple demo without external services."""
    print("\n" + "=" * 60)
    print("  IN-VEHICLE AI COPILOT - SIMPLE DEMO")
    print("=" * 60)
    print("\nThis demo shows the warning system with simulated detections.\n")

    time.sleep(1)

    # Scenario 1: Clear road
    print("[1/3] Scanning road ahead...")
    time.sleep(1.5)
    trigger_warning("none", "Road clear. Safe to proceed.")

    time.sleep(2)

    # Scenario 2: Pedestrian detected
    print("\n[2/3] Scanning road ahead...")
    time.sleep(1.5)
    trigger_warning("high", "CAUTION: Pedestrian detected ahead! Reduce speed.")

    time.sleep(2)

    # Scenario 3: Multiple pedestrians
    print("\n[3/3] Scanning road ahead...")
    time.sleep(1.5)
    trigger_warning("critical", "EMERGENCY: Multiple pedestrians crossing! Prepare to stop.")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60 + "\n")


def demo_with_agent(use_ollama: bool = True):
    """Demo with agent reasoning."""
    print("\n" + "=" * 60)
    print("  IN-VEHICLE AI COPILOT - AGENT DEMO")
    print("=" * 60)

    if use_ollama:
        print("\nUsing Ollama (gemma3:4b) for intelligent reasoning...")
    else:
        print("\nUsing rule-based fallback (Ollama not enabled)...")

    pipeline = SafetyPipeline(
        use_ollama=use_ollama,
        enable_visual=True,
        enable_audio=False
    )

    time.sleep(1)

    # Test scenarios
    scenarios = [
        (False, "Clear highway, no obstacles"),
        (True, "Pedestrian crossing at intersection"),
        (True, "Pedestrian on sidewalk near road"),
    ]

    for i, (detected, description) in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Scenario: {description}")
        print("-" * 40)
        time.sleep(1)

        result = pipeline.run_local(detected=detected)

        print(f"  Detection: {'PEDESTRIAN' if detected else 'CLEAR'}")
        print(f"  Alert Level: {result.decision.alert_level}")
        print(f"  Action: {result.decision.action}")
        if result.decision.reasoning:
            print(f"  Reasoning: {result.decision.reasoning}")
        print(f"  Time: {result.total_time_ms:.0f}ms")

        time.sleep(2)

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60 + "\n")


def demo_with_server(server_url: str, image_path: str):
    """Demo with detection server."""
    import requests

    print("\n" + "=" * 60)
    print("  IN-VEHICLE AI COPILOT - FULL PIPELINE DEMO")
    print("=" * 60)
    print(f"\nDetection Server: {server_url}")
    print(f"Test Image: {image_path}\n")

    # Check server health
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        health = resp.json()
        print(f"Server Status: {health['status']}")
        print(f"Model Loaded: {health['model_loaded']}")
        print(f"GPU: {health.get('gpu_name', 'N/A')}")
    except Exception as e:
        print(f"Cannot connect to server: {e}")
        return

    print("-" * 40)

    # Create pipeline
    pipeline = SafetyPipeline(
        detection_url=server_url,
        use_ollama=False,  # Use rule-based for demo
        enable_visual=True
    )

    # Run detection
    print(f"\nProcessing image...")
    result = pipeline.run(image_path)

    print(f"\nResults:")
    print(f"  Pedestrian Detected: {result.pedestrian_detected}")
    print(f"  Alert Level: {result.decision.alert_level}")
    print(f"  Message: {result.decision.message}")
    print(f"  Detection Time: {result.detection_time_ms:.0f}ms")
    print(f"  Total Time: {result.total_time_ms:.0f}ms")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="In-Vehicle AI Copilot Demo")
    parser.add_argument("--simple", action="store_true",
                       help="Run simple warning demo")
    parser.add_argument("--no-ollama", action="store_true",
                       help="Use rule-based agent instead of Ollama")
    parser.add_argument("--detection-server", type=str,
                       help="URL of detection server (e.g., http://VM_IP:8000)")
    parser.add_argument("--image", type=str,
                       help="Path to test image (for server demo)")
    args = parser.parse_args()

    if args.simple:
        demo_simple()
    elif args.detection_server:
        if not args.image:
            print("Error: --image required with --detection-server")
            return
        demo_with_server(args.detection_server, args.image)
    else:
        demo_with_agent(use_ollama=not args.no_ollama)


if __name__ == "__main__":
    main()
