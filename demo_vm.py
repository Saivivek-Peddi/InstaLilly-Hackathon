#!/usr/bin/env python3
"""
Full Agentic Pipeline Demo - Runs on VM

This demo shows the complete flow:
1. Image -> PaliGemma Detection -> "yes/no"
2. Detection -> Ollama (gemma3n) Agent -> Decision
3. Decision -> Warning Display

Usage:
    python demo_vm.py
    python demo_vm.py --image path/to/image.jpg
"""

import sys
import time
import json
import requests
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).parent
SAMPLE_PEDESTRIAN = PROJECT_DIR / "models/pedestrian_detector/data/penn_fudan/PNGImages/FudanPed00001.png"
SAMPLE_NO_PED = PROJECT_DIR / "models/pedestrian_detector/data/processed/images/kitti_no_pedestrian_007254.jpg"

# Detection server (local since we're on VM)
DETECTION_URL = "http://localhost:8000"

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3n"

SYSTEM_PROMPT = """You are an in-vehicle AI safety copilot. Your job is to assess threats and provide clear, concise warnings to the driver.

When given detection results, you must:
1. Assess the threat level (none/low/medium/high/critical)
2. Decide on an action (none/alert/warn/emergency)
3. Generate a brief warning message for the driver (max 15 words)

Respond in JSON format only:
{"alert_level": "high", "action": "warn", "message": "Pedestrian detected ahead. Reduce speed.", "reasoning": "Single pedestrian in path requires caution"}"""


def detect_pedestrian(image_path: str) -> dict:
    """Call detection server."""
    print(f"   Sending image to PaliGemma detector...")
    start = time.time()

    with open(image_path, "rb") as f:
        response = requests.post(
            f"{DETECTION_URL}/detect",
            files={"image": f},
            timeout=60
        )

    elapsed = (time.time() - start) * 1000
    result = response.json()
    result["detection_time_ms"] = elapsed
    return result


def call_ollama_agent(detected: bool) -> dict:
    """Call Ollama for agentic reasoning."""
    print(f"   Sending to Gemma3n agent for reasoning...")
    start = time.time()

    prompt = f"""Detection Result:
- Pedestrian Detected: {detected}
- Confidence: 100%
- Source: PaliGemma pedestrian detector

Analyze the situation and provide your decision in JSON format."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 200
                }
            },
            timeout=60
        )

        elapsed = (time.time() - start) * 1000

        if response.status_code == 200:
            text = response.json().get("response", "")
            # Parse JSON from response
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                decision = json.loads(text[start_idx:end_idx])
                decision["agent_time_ms"] = elapsed
                return decision
    except Exception as e:
        print(f"   Agent error: {e}")

    # Fallback
    return {
        "alert_level": "high" if detected else "none",
        "action": "warn" if detected else "none",
        "message": "CAUTION: Pedestrian detected!" if detected else "Road clear.",
        "reasoning": "Fallback rule-based decision",
        "agent_time_ms": 0
    }


def speak_warning(message: str, save_file: str = "/tmp/warning.wav"):
    """Speak the warning using text-to-speech and save to file."""
    import subprocess
    try:
        # Generate WAV file using espeak-ng
        subprocess.run(
            ["espeak-ng", "-v", "en", "-s", "150", "-w", save_file, message],
            capture_output=True,
            timeout=10
        )
        print(f"   Audio saved to: {save_file}")
    except Exception as e:
        print(f"   (TTS unavailable: {e})")


def display_warning(decision: dict, enable_voice: bool = True):
    """Display the warning with formatting and optional voice."""
    level = decision.get("alert_level", "none")
    message = decision.get("message", "")

    # Color codes
    colors = {
        "none": "\033[92m",      # Green
        "low": "\033[93m",       # Yellow
        "medium": "\033[93m",    # Yellow
        "high": "\033[91m",      # Red
        "critical": "\033[91m",  # Red
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    icons = {
        "none": "✓",
        "low": "ℹ",
        "medium": "⚠",
        "high": "⚠",
        "critical": "🚨"
    }

    color = colors.get(level, RESET)
    icon = icons.get(level, "•")

    if level == "none":
        print(f"\n{color}   {icon} {message}{RESET}")
    else:
        print(f"\n{color}{BOLD}")
        print("   " + "=" * 56)
        print(f"     {icon}  {level.upper()} ALERT  {icon}")
        print("   " + "=" * 56)
        print(f"     {message}")
        print("   " + "=" * 56)
        print(f"{RESET}")

    # Speak the warning if enabled and not "none" level
    if enable_voice and level != "none":
        print("   🔊 Speaking warning...")
        speak_warning(message)


def run_pipeline(image_path: str):
    """Run the full agentic pipeline."""
    print(f"\n{'='*60}")
    print(f"   Processing: {Path(image_path).name}")
    print(f"{'='*60}")

    # Step 1: Detection
    print("\n[1/3] DETECTION (PaliGemma + LoRA)")
    detection = detect_pedestrian(image_path)
    detected = detection.get("pedestrian_detected", False)
    print(f"   Result: {'PEDESTRIAN DETECTED' if detected else 'NO PEDESTRIAN'}")
    print(f"   Raw response: {detection.get('raw_response', 'N/A')}")
    print(f"   Time: {detection.get('detection_time_ms', 0):.0f}ms")

    # Step 2: Agent reasoning
    print("\n[2/3] AGENT REASONING (Gemma3n via Ollama)")
    decision = call_ollama_agent(detected)
    print(f"   Alert Level: {decision.get('alert_level', 'N/A')}")
    print(f"   Action: {decision.get('action', 'N/A')}")
    print(f"   Reasoning: {decision.get('reasoning', 'N/A')}")
    print(f"   Time: {decision.get('agent_time_ms', 0):.0f}ms")

    # Step 3: Warning
    print("\n[3/3] WARNING OUTPUT")
    display_warning(decision)

    # Summary
    total_time = detection.get("detection_time_ms", 0) + decision.get("agent_time_ms", 0)
    print(f"\n   Total pipeline time: {total_time:.0f}ms")

    return detection, decision


def main():
    print("\n" + "=" * 60)
    print("   IN-VEHICLE AI COPILOT - FULL AGENTIC PIPELINE")
    print("=" * 60)

    # Check services
    print("\nChecking services...")

    # Check detection server
    try:
        health = requests.get(f"{DETECTION_URL}/health", timeout=5).json()
        print(f"   Detection Server: ✅ {health.get('status', 'unknown')}")
        print(f"   GPU: {health.get('gpu_name', 'N/A')}")
    except:
        print("   Detection Server: ❌ Not running")
        print("   Start with: uvicorn api.detection_server:app --port 8000")
        return

    # Check Ollama
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if OLLAMA_MODEL in str(models):
            print(f"   Ollama ({OLLAMA_MODEL}): ✅ Ready")
        else:
            print(f"   Ollama: ⚠ Model {OLLAMA_MODEL} not found")
            print(f"   Available: {models}")
    except:
        print("   Ollama: ❌ Not running")
        return

    print("\n" + "-" * 60)

    # Get image path from args or use samples
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        run_pipeline(image_path)
    else:
        # Run both test cases
        print("\n📸 TEST 1: Image WITH pedestrians")
        if SAMPLE_PEDESTRIAN.exists():
            run_pipeline(str(SAMPLE_PEDESTRIAN))
        else:
            print(f"   Sample not found: {SAMPLE_PEDESTRIAN}")

        time.sleep(2)

        print("\n📸 TEST 2: Image WITHOUT pedestrians")
        if SAMPLE_NO_PED.exists():
            run_pipeline(str(SAMPLE_NO_PED))
        else:
            print(f"   Sample not found: {SAMPLE_NO_PED}")

    print("\n" + "=" * 60)
    print("   DEMO COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
