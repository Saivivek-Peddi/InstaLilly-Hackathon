#!/usr/bin/env python3
"""
Unified Router for In-Vehicle AI Copilot

Input: JSON with camera sources
Output: TTS warning

Example:
{
    "internal_camera": "/path/to/driver_image.jpg",
    "external_camera": "/path/to/road_image.jpg"
}
"""

import json
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import torch
from PIL import Image


@dataclass
class DetectionResults:
    """Combined results from all detectors."""
    # Drowsiness (internal camera)
    driver_state: str = "alert"  # alert, drowsy, yawning, eyes_closed

    # Distraction (internal camera)
    driver_activity: str = "safe_driving"  # safe_driving, texting_phone, talking_phone, other_activities, turning

    # Pedestrian (external camera)
    pedestrian_detected: bool = False

    # Timing
    drowsiness_time_ms: float = 0
    distraction_time_ms: float = 0
    pedestrian_time_ms: float = 0


@dataclass
class AgentDecision:
    """Decision from the agent."""
    alert_level: str  # none, low, medium, high, critical
    message: str
    reasoning: str


class ModelRouter:
    """
    Routes inputs to appropriate models and combines results.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.models_loaded = False

        # Shared base model and processor
        self.base_model = None
        self.processor = None
        self.peft_model = None
        self.current_adapter = None
        self.checkpoints = {}

    def load_models(self):
        """Load all fine-tuned models with shared base for memory efficiency."""
        if self.models_loaded:
            return

        print("Loading models...")
        from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
        from peft import PeftModel

        BASE_MODEL = "google/paligemma2-3b-pt-448"

        # Paths to fine-tuned checkpoints
        PROJECT_DIR = Path(__file__).parent.parent
        DROWSINESS_CKPT = PROJECT_DIR / "models/drowsiness_detector/checkpoints/final"
        DISTRACTION_CKPT = PROJECT_DIR / "models/distraction_detector/checkpoints/final"
        PEDESTRIAN_CKPT = PROJECT_DIR / "models/pedestrian_detector/checkpoints/final"

        # 4-bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load base processor (shared)
        print("  Loading processor...")
        self.processor = PaliGemmaProcessor.from_pretrained(BASE_MODEL)

        # Load single base model with quantization
        print("  Loading base model (4-bit)...")
        self.base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        # Store checkpoint paths - will load LoRA adapters on demand
        self.checkpoints = {
            "drowsiness": DROWSINESS_CKPT if DROWSINESS_CKPT.exists() else None,
            "distraction": DISTRACTION_CKPT if DISTRACTION_CKPT.exists() else None,
            "pedestrian": PEDESTRIAN_CKPT if PEDESTRIAN_CKPT.exists() else None,
        }

        # Log available models
        for name, path in self.checkpoints.items():
            if path:
                print(f"  Found {name} checkpoint: {path}")
            else:
                print(f"  Warning: {name} checkpoint not found")

        # Current loaded adapter
        self.current_adapter = None
        self.peft_model = None

        self.models_loaded = True
        print("Models loaded!")

    def _switch_adapter(self, adapter_name: str):
        """Switch to a different LoRA adapter."""
        from peft import PeftModel

        if self.current_adapter == adapter_name:
            return  # Already loaded

        checkpoint_path = self.checkpoints.get(adapter_name)
        if not checkpoint_path:
            return

        print(f"  Switching to {adapter_name} adapter...")

        # Load new adapter
        self.peft_model = PeftModel.from_pretrained(
            self.base_model,
            str(checkpoint_path)
        )
        self.peft_model.eval()
        self.current_adapter = adapter_name

    def _run_inference(self, adapter_name: str, image_path: str, prompt: str) -> str:
        """Run inference with specified adapter."""
        self._switch_adapter(adapter_name)

        if not self.peft_model:
            return ""

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        # Extract answer after prompt
        answer = response.split("?")[-1].strip().lower()
        return answer

    def detect_drowsiness(self, image_path: str) -> tuple[str, float]:
        """Detect driver drowsiness state."""
        if not self.checkpoints.get("drowsiness"):
            return "alert", 0

        start = time.time()
        answer = self._run_inference(
            "drowsiness",
            image_path,
            "Driver state?\n"
        )
        elapsed = (time.time() - start) * 1000

        # Map to class
        classes = ["alert", "drowsy", "yawning", "eyes_closed"]
        for cls in classes:
            if cls in answer:
                return cls, elapsed
        return "alert", elapsed

    def detect_distraction(self, image_path: str) -> tuple[str, float]:
        """Detect driver distraction activity."""
        if not self.checkpoints.get("distraction"):
            return "safe_driving", 0

        start = time.time()
        answer = self._run_inference(
            "distraction",
            image_path,
            "Driver activity?\n"
        )
        elapsed = (time.time() - start) * 1000

        # Map to class
        classes = ["safe_driving", "texting_phone", "talking_phone", "other_activities", "turning"]
        for cls in classes:
            if cls in answer or cls.replace("_", " ") in answer:
                return cls, elapsed
        return "safe_driving", elapsed

    def detect_pedestrian(self, image_path: str) -> tuple[bool, float]:
        """Detect pedestrians in road view."""
        if not self.checkpoints.get("pedestrian"):
            return False, 0

        start = time.time()
        answer = self._run_inference(
            "pedestrian",
            image_path,
            "Pedestrian?\n"
        )
        elapsed = (time.time() - start) * 1000

        detected = "yes" in answer
        return detected, elapsed

    def process(self, input_json: Dict[str, Any]) -> DetectionResults:
        """
        Process input JSON and route to appropriate models.

        Args:
            input_json: {
                "internal_camera": "/path/to/image.jpg",
                "external_camera": "/path/to/image.jpg"
            }

        Returns:
            DetectionResults with all detection outputs
        """
        self.load_models()

        results = DetectionResults()

        # Process internal camera (drowsiness + distraction)
        if "internal_camera" in input_json and input_json["internal_camera"]:
            internal_path = input_json["internal_camera"]

            # Drowsiness detection
            results.driver_state, results.drowsiness_time_ms = self.detect_drowsiness(internal_path)

            # Distraction detection
            results.driver_activity, results.distraction_time_ms = self.detect_distraction(internal_path)

        # Process external camera (pedestrian)
        if "external_camera" in input_json and input_json["external_camera"]:
            external_path = input_json["external_camera"]
            results.pedestrian_detected, results.pedestrian_time_ms = self.detect_pedestrian(external_path)

        return results


class SafetyAgent:
    """
    Agent that reasons about combined detections and generates warnings.
    """

    def decide(self, results: DetectionResults) -> AgentDecision:
        """
        Make decision based on all detection results.
        Uses rule-based logic for speed (can be replaced with LLM).
        """
        alerts = []
        severity = 0

        # Check drowsiness (internal camera)
        if results.driver_state == "eyes_closed":
            alerts.append("Your eyes are closed")
            severity = max(severity, 4)  # Critical
        elif results.driver_state == "drowsy":
            alerts.append("You appear drowsy")
            severity = max(severity, 3)  # High
        elif results.driver_state == "yawning":
            alerts.append("You are yawning")
            severity = max(severity, 2)  # Medium

        # Check distraction (internal camera)
        if results.driver_activity == "texting_phone":
            alerts.append("Stop texting while driving")
            severity = max(severity, 4)  # Critical
        elif results.driver_activity == "talking_phone":
            alerts.append("Phone call detected")
            severity = max(severity, 2)  # Medium
        elif results.driver_activity == "other_activities":
            alerts.append("Focus on the road")
            severity = max(severity, 2)  # Medium

        # Check pedestrian (external camera)
        if results.pedestrian_detected:
            alerts.append("Pedestrian ahead")
            severity = max(severity, 3)  # High

        # Combine alerts
        if not alerts:
            return AgentDecision(
                alert_level="none",
                message="All clear. Drive safely.",
                reasoning="No hazards detected"
            )

        # Map severity to level
        levels = {0: "none", 1: "low", 2: "medium", 3: "high", 4: "critical"}
        level = levels.get(severity, "high")

        # Build message
        if len(alerts) == 1:
            message = f"Warning: {alerts[0]}!"
        else:
            message = f"ALERT: {' and '.join(alerts)}!"

        # Escalate if multiple issues
        if len(alerts) >= 2 and severity < 4:
            level = "critical"
            message = f"CRITICAL: {' and '.join(alerts)}!"

        return AgentDecision(
            alert_level=level,
            message=message,
            reasoning=f"Detected: {', '.join(alerts)}"
        )


class TTSOutput:
    """Text-to-speech output."""

    @staticmethod
    def speak(message: str, output_file: str = "/tmp/warning.wav") -> str:
        """
        Convert message to speech and save as audio file.

        Returns:
            Path to audio file
        """
        try:
            # Try espeak-ng first (Linux)
            subprocess.run(
                ["espeak-ng", "-v", "en", "-s", "150", "-w", output_file, message],
                capture_output=True,
                timeout=10
            )
            return output_file
        except FileNotFoundError:
            try:
                # Try espeak (fallback)
                subprocess.run(
                    ["espeak", "-v", "en", "-s", "150", "-w", output_file, message],
                    capture_output=True,
                    timeout=10
                )
                return output_file
            except FileNotFoundError:
                print(f"TTS not available. Message: {message}")
                return ""

    @staticmethod
    def play(audio_file: str):
        """Play audio file."""
        try:
            subprocess.run(["aplay", audio_file], capture_output=True, timeout=10)
        except:
            pass


class InVehicleCopilot:
    """
    Main class that orchestrates the full pipeline.

    Input JSON → Router → Models → Agent → TTS
    """

    def __init__(self):
        self.router = ModelRouter()
        self.agent = SafetyAgent()
        self.tts = TTSOutput()

    def process(self, input_json: Dict[str, Any], speak: bool = True) -> Dict[str, Any]:
        """
        Process input and generate warning.

        Args:
            input_json: {
                "internal_camera": "/path/to/driver.jpg",
                "external_camera": "/path/to/road.jpg"
            }
            speak: Whether to speak the warning

        Returns:
            {
                "detections": {...},
                "decision": {...},
                "audio_file": "/tmp/warning.wav",
                "total_time_ms": 123.4
            }
        """
        start = time.time()

        # Step 1: Route and detect
        results = self.router.process(input_json)

        # Step 2: Agent reasoning
        decision = self.agent.decide(results)

        # Step 3: TTS output
        audio_file = ""
        if speak and decision.alert_level != "none":
            audio_file = self.tts.speak(decision.message)

        total_time = (time.time() - start) * 1000

        return {
            "detections": asdict(results),
            "decision": asdict(decision),
            "audio_file": audio_file,
            "total_time_ms": total_time
        }

    def run_demo(self, internal_image: str = None, external_image: str = None):
        """Run a demo with sample images."""
        print("\n" + "=" * 60)
        print("  IN-VEHICLE AI COPILOT")
        print("=" * 60)

        input_json = {}
        if internal_image:
            input_json["internal_camera"] = internal_image
        if external_image:
            input_json["external_camera"] = external_image

        print(f"\nInput: {json.dumps(input_json, indent=2)}")
        print("-" * 60)

        result = self.process(input_json)

        print(f"\nDetections:")
        print(f"  Driver State: {result['detections']['driver_state']}")
        print(f"  Driver Activity: {result['detections']['driver_activity']}")
        print(f"  Pedestrian: {'Yes' if result['detections']['pedestrian_detected'] else 'No'}")

        print(f"\nDecision:")
        print(f"  Alert Level: {result['decision']['alert_level'].upper()}")
        print(f"  Message: {result['decision']['message']}")

        print(f"\nTiming:")
        print(f"  Drowsiness: {result['detections']['drowsiness_time_ms']:.0f}ms")
        print(f"  Distraction: {result['detections']['distraction_time_ms']:.0f}ms")
        print(f"  Pedestrian: {result['detections']['pedestrian_time_ms']:.0f}ms")
        print(f"  Total: {result['total_time_ms']:.0f}ms")

        if result['audio_file']:
            print(f"\nAudio: {result['audio_file']}")

        print("=" * 60)

        return result


if __name__ == "__main__":
    import sys

    copilot = InVehicleCopilot()

    if len(sys.argv) >= 3:
        # Usage: python router.py <internal_image> <external_image>
        copilot.run_demo(
            internal_image=sys.argv[1],
            external_image=sys.argv[2]
        )
    elif len(sys.argv) == 2:
        # Single image - assume internal camera
        copilot.run_demo(internal_image=sys.argv[1])
    else:
        print("Usage: python router.py <internal_camera_image> [external_camera_image]")
        print("\nExample:")
        print("  python router.py driver.jpg road.jpg")
