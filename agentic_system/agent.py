"""
Agentic Decision Agent using Ollama

This agent receives detection results and decides on appropriate actions
using Gemma via Ollama for reasoning.
"""

import json
import requests
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class AlertLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectionResult:
    """Result from pedestrian detection model."""
    detected: bool
    confidence: float = 1.0
    count: int = 0
    source: str = "pedestrian_detector"


@dataclass
class AgentDecision:
    """Decision made by the agent."""
    alert_level: AlertLevel
    action: str
    message: str
    reasoning: str


class SafetyAgent:
    """
    Agentic safety system that processes detections and decides on actions.
    Uses Ollama with Gemma for intelligent reasoning.
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL = "gemma3n"  # Gemma 3n for on-device inference

    SYSTEM_PROMPT = """You are an in-vehicle AI safety copilot. Your job is to assess threats and provide clear, concise warnings to the driver.

When given detection results, you must:
1. Assess the threat level (none/low/medium/high/critical)
2. Decide on an action (none/alert/warn/emergency)
3. Generate a brief warning message for the driver (max 15 words)

Respond in JSON format:
{
    "alert_level": "high",
    "action": "warn",
    "message": "Pedestrian detected ahead. Reduce speed.",
    "reasoning": "Single pedestrian in path requires caution"
}"""

    def __init__(self, ollama_url: Optional[str] = None, model: Optional[str] = None):
        self.ollama_url = ollama_url or self.OLLAMA_URL
        self.model = model or self.MODEL
        self._check_ollama()

    def _check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            print("Warning: Ollama not running. Start with: ollama serve")
            return False

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for reasoning."""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": self.SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temp for consistent decisions
                        "num_predict": 200
                    }
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return ""
        except Exception as e:
            print(f"Ollama error: {e}")
            return ""

    def _parse_response(self, response: str) -> AgentDecision:
        """Parse Ollama response into AgentDecision."""
        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return AgentDecision(
                    alert_level=AlertLevel(data.get("alert_level", "none")),
                    action=data.get("action", "none"),
                    message=data.get("message", ""),
                    reasoning=data.get("reasoning", "")
                )
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Parse error: {e}")

        # Fallback decision
        return AgentDecision(
            alert_level=AlertLevel.NONE,
            action="none",
            message="",
            reasoning="Failed to parse agent response"
        )

    def decide(self, detection: DetectionResult, context: Optional[dict] = None) -> AgentDecision:
        """
        Make a decision based on detection results.

        Args:
            detection: Result from detection model
            context: Optional context (speed, location, etc.)

        Returns:
            AgentDecision with alert level, action, and message
        """
        # Build prompt
        prompt = f"""Detection Result:
- Pedestrian Detected: {detection.detected}
- Confidence: {detection.confidence:.0%}
- Source: {detection.source}
"""
        if context:
            prompt += f"\nVehicle Context:\n"
            for k, v in context.items():
                prompt += f"- {k}: {v}\n"

        prompt += "\nAnalyze the situation and provide your decision in JSON format."

        # Get agent reasoning
        response = self._call_ollama(prompt)

        if not response:
            # Fallback: rule-based decision if Ollama fails
            return self._fallback_decision(detection)

        return self._parse_response(response)

    def _fallback_decision(self, detection: DetectionResult) -> AgentDecision:
        """Rule-based fallback when Ollama is unavailable."""
        if not detection.detected:
            return AgentDecision(
                alert_level=AlertLevel.NONE,
                action="none",
                message="Road clear. No pedestrians detected.",
                reasoning="No detection, no action needed"
            )

        # Pedestrian detected
        return AgentDecision(
            alert_level=AlertLevel.HIGH,
            action="warn",
            message="CAUTION: Pedestrian detected ahead!",
            reasoning="Pedestrian detected, warning driver"
        )

    def process_frame(self, pedestrian_detected: bool, confidence: float = 1.0) -> AgentDecision:
        """
        Convenience method to process a single frame detection.

        Args:
            pedestrian_detected: Whether pedestrian was detected
            confidence: Detection confidence (0-1)

        Returns:
            AgentDecision
        """
        detection = DetectionResult(
            detected=pedestrian_detected,
            confidence=confidence,
            source="pedestrian_detector"
        )
        return self.decide(detection)


# Simple function-based interface
def get_warning(pedestrian_detected: bool) -> str:
    """
    Simple interface to get a warning message.

    Args:
        pedestrian_detected: Whether pedestrian was detected

    Returns:
        Warning message string
    """
    agent = SafetyAgent()
    decision = agent.process_frame(pedestrian_detected)
    return decision.message


if __name__ == "__main__":
    # Test the agent
    print("Testing Safety Agent...")
    print("-" * 50)

    agent = SafetyAgent()

    # Test 1: Pedestrian detected
    print("\nTest 1: Pedestrian Detected")
    decision = agent.process_frame(pedestrian_detected=True)
    print(f"  Alert Level: {decision.alert_level.value}")
    print(f"  Action: {decision.action}")
    print(f"  Message: {decision.message}")
    print(f"  Reasoning: {decision.reasoning}")

    # Test 2: No pedestrian
    print("\nTest 2: No Pedestrian")
    decision = agent.process_frame(pedestrian_detected=False)
    print(f"  Alert Level: {decision.alert_level.value}")
    print(f"  Action: {decision.action}")
    print(f"  Message: {decision.message}")
    print(f"  Reasoning: {decision.reasoning}")
