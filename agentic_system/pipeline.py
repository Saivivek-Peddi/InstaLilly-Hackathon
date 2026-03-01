"""
End-to-End Safety Pipeline

Orchestrates: Detection -> Agent -> Actions
"""

import time
import requests
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

from .agent import SafetyAgent, DetectionResult, AgentDecision
from .actions import AlertExecutor


@dataclass
class PipelineResult:
    """Result from full pipeline execution."""
    pedestrian_detected: bool
    decision: AgentDecision
    detection_time_ms: float
    agent_time_ms: float
    total_time_ms: float


class SafetyPipeline:
    """
    Complete safety detection and alert pipeline.

    Flow:
    1. Image -> Detection Server (PaliGemma) -> yes/no
    2. Detection -> Agent (Ollama/Gemma) -> Decision
    3. Decision -> Alert Executor -> Warning
    """

    def __init__(
        self,
        detection_url: str = "http://localhost:8000",
        enable_audio: bool = False,
        enable_visual: bool = True,
        use_ollama: bool = True
    ):
        self.detection_url = detection_url
        self.agent = SafetyAgent() if use_ollama else None
        self.executor = AlertExecutor(
            enable_text=True,
            enable_audio=enable_audio,
            enable_visual=enable_visual
        )
        self.use_ollama = use_ollama

    def detect_pedestrian(self, image_path: str) -> tuple[bool, float]:
        """
        Call detection server to check for pedestrians.

        Args:
            image_path: Path to image file

        Returns:
            (detected: bool, time_ms: float)
        """
        start = time.time()

        try:
            with open(image_path, "rb") as f:
                response = requests.post(
                    f"{self.detection_url}/detect",
                    files={"image": f},
                    timeout=30
                )

            if response.status_code == 200:
                result = response.json()
                detected = result.get("pedestrian_detected", False)
            else:
                print(f"Detection error: {response.status_code}")
                detected = False

        except requests.exceptions.ConnectionError:
            print("Warning: Detection server not running")
            detected = False
        except Exception as e:
            print(f"Detection error: {e}")
            detected = False

        elapsed = (time.time() - start) * 1000
        return detected, elapsed

    def process_detection(self, detected: bool) -> tuple[AgentDecision, float]:
        """
        Process detection through agent.

        Args:
            detected: Whether pedestrian was detected

        Returns:
            (decision: AgentDecision, time_ms: float)
        """
        start = time.time()

        if self.use_ollama and self.agent:
            decision = self.agent.process_frame(
                pedestrian_detected=detected
            )
        else:
            # Simple rule-based fallback
            if detected:
                decision = AgentDecision(
                    alert_level="high",
                    action="warn",
                    message="CAUTION: Pedestrian detected ahead!",
                    reasoning="Rule-based: pedestrian detected"
                )
            else:
                decision = AgentDecision(
                    alert_level="none",
                    action="none",
                    message="Road clear.",
                    reasoning="Rule-based: no detection"
                )

        elapsed = (time.time() - start) * 1000
        return decision, elapsed

    def execute_action(self, decision: AgentDecision):
        """Execute alert based on decision."""
        level = decision.alert_level
        if hasattr(level, 'value'):
            level = level.value
        self.executor.execute(level, decision.message)

    def run(self, image_path: str, execute_alert: bool = True) -> PipelineResult:
        """
        Run full pipeline on an image.

        Args:
            image_path: Path to image
            execute_alert: Whether to execute the alert

        Returns:
            PipelineResult
        """
        total_start = time.time()

        # Step 1: Detection
        detected, detection_time = self.detect_pedestrian(image_path)

        # Step 2: Agent decision
        decision, agent_time = self.process_detection(detected)

        # Step 3: Execute alert
        if execute_alert:
            self.execute_action(decision)

        total_time = (time.time() - total_start) * 1000

        return PipelineResult(
            pedestrian_detected=detected,
            decision=decision,
            detection_time_ms=detection_time,
            agent_time_ms=agent_time,
            total_time_ms=total_time
        )

    def run_local(self, detected: bool, execute_alert: bool = True) -> PipelineResult:
        """
        Run pipeline with pre-computed detection result.
        Useful when detection runs on remote server.

        Args:
            detected: Pre-computed detection result
            execute_alert: Whether to execute the alert

        Returns:
            PipelineResult
        """
        total_start = time.time()

        # Skip detection, use provided result
        decision, agent_time = self.process_detection(detected)

        if execute_alert:
            self.execute_action(decision)

        total_time = (time.time() - total_start) * 1000

        return PipelineResult(
            pedestrian_detected=detected,
            decision=decision,
            detection_time_ms=0,
            agent_time_ms=agent_time,
            total_time_ms=total_time
        )


def demo_pipeline():
    """Demo the pipeline with simulated detections."""
    print("=" * 60)
    print("  IN-VEHICLE AI COPILOT - SAFETY PIPELINE DEMO")
    print("=" * 60)

    # Create pipeline (without Ollama for demo)
    pipeline = SafetyPipeline(use_ollama=False, enable_visual=True)

    print("\n[Scenario 1] No pedestrian detected...")
    time.sleep(1)
    result = pipeline.run_local(detected=False)
    print(f"  Detection: {result.pedestrian_detected}")
    print(f"  Decision: {result.decision.action}")
    print(f"  Time: {result.total_time_ms:.1f}ms")

    time.sleep(2)

    print("\n[Scenario 2] Pedestrian detected!")
    time.sleep(1)
    result = pipeline.run_local(detected=True)
    print(f"  Detection: {result.pedestrian_detected}")
    print(f"  Decision: {result.decision.action}")
    print(f"  Time: {result.total_time_ms:.1f}ms")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo_pipeline()
