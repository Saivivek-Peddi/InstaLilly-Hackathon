#!/usr/bin/env python3
"""
Test the In-Vehicle AI Copilot API

Prerequisites:
1. Start API on VM with port forwarding:
   ssh -L 8000:localhost:8000 hackathon@34.123.38.202 \
     "cd ~/InstaLilly-Hackathon && uv run uvicorn api.copilot_server:app --port 8000"

2. Run this script:
   python test_api.py
"""

import base64
import requests
import json
from pathlib import Path

API_URL = "http://localhost:8000"

# Test images
TEST_IMAGES = {
    "drowsy_driver": "test_images/drowsy_driver.jpg",
    "alert_driver": "test_images/alert_driver.jpg",
    "pedestrian": "test_images/pedestrian.png",
    "no_pedestrian": "test_images/no_pedestrian.png",
}


def encode_image(path: str) -> str:
    """Encode image to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_api(internal: str = None, external: str = None) -> dict:
    """Call the copilot API."""
    payload = {}

    if internal:
        payload["internal_camera"] = f"data:image/jpeg;base64,{encode_image(internal)}"
    if external:
        payload["external_camera"] = f"data:image/png;base64,{encode_image(external)}"

    response = requests.post(
        f"{API_URL}/process",
        json=payload,
        timeout=120
    )

    result = response.json()
    if "detail" in result:
        print(f"  API Error: {result['detail']}")
        return None
    return result


def print_result(name: str, result: dict):
    """Pretty print result."""
    if result is None:
        print(f"  [ERROR] No result returned")
        return

    if "decision" not in result:
        print(f"  [ERROR] Unexpected response: {result}")
        return

    colors = {
        "none": "\033[92m",      # Green
        "low": "\033[93m",       # Yellow
        "medium": "\033[93m",    # Yellow
        "high": "\033[91m",      # Red
        "critical": "\033[91m",  # Red
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    level = result["decision"]["alert_level"]
    color = colors.get(level, RESET)

    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")
    print(f"  Driver State:    {result['detections']['driver_state']}")
    print(f"  Driver Activity: {result['detections']['driver_activity']}")
    print(f"  Pedestrian:      {'YES' if result['detections']['pedestrian_detected'] else 'NO'}")
    print(f"  Alert Level:     {color}{BOLD}{level.upper()}{RESET}")
    print(f"  Message:         {color}{result['decision']['message']}{RESET}")
    print(f"  Time:            {result['total_time_ms']:.0f}ms")


def main():
    print("\n" + "="*60)
    print("  IN-VEHICLE AI COPILOT - API TEST")
    print("="*60)

    # Check API health
    print("\nChecking API health...")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        print(f"  Status: {health['status']}")
        print(f"  GPU: {health.get('gpu_name', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")
        print("\nMake sure the API is running with port forwarding:")
        print("  ssh -L 8000:localhost:8000 hackathon@34.123.38.202 \\")
        print('    "cd ~/InstaLilly-Hackathon && uv run uvicorn api.copilot_server:app --port 8000"')
        return

    # Test 1: Alert driver, no pedestrian (should be NONE)
    print("\n" + "-"*60)
    print("Test 1: Alert driver, no pedestrian")
    result = call_api(
        internal=TEST_IMAGES["alert_driver"],
        external=TEST_IMAGES["no_pedestrian"]
    )
    print_result("Alert + No Pedestrian", result)

    # Test 2: Alert driver, pedestrian detected (should be HIGH)
    print("\n" + "-"*60)
    print("Test 2: Alert driver, pedestrian ahead")
    result = call_api(
        internal=TEST_IMAGES["alert_driver"],
        external=TEST_IMAGES["pedestrian"]
    )
    print_result("Alert + Pedestrian", result)

    # Test 3: Drowsy driver, no pedestrian (should be HIGH)
    print("\n" + "-"*60)
    print("Test 3: Drowsy driver, no pedestrian")
    result = call_api(
        internal=TEST_IMAGES["drowsy_driver"],
        external=TEST_IMAGES["no_pedestrian"]
    )
    print_result("Drowsy + No Pedestrian", result)

    # Test 4: Drowsy driver + pedestrian (should be CRITICAL)
    print("\n" + "-"*60)
    print("Test 4: Drowsy driver + pedestrian (CRITICAL)")
    result = call_api(
        internal=TEST_IMAGES["drowsy_driver"],
        external=TEST_IMAGES["pedestrian"]
    )
    print_result("Drowsy + Pedestrian", result)

    # Test 5: Only external camera
    print("\n" + "-"*60)
    print("Test 5: External camera only (pedestrian)")
    result = call_api(external=TEST_IMAGES["pedestrian"])
    print_result("Pedestrian Only", result)

    print("\n" + "="*60)
    print("  ALL TESTS COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
