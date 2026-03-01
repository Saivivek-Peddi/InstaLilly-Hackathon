"""
Agentic System for In-Vehicle AI Copilot

Components:
- router.py: Main router (JSON input -> Models -> TTS output)
- agent.py: Ollama-based decision agent
- actions.py: Alert/warning executors
- pipeline.py: End-to-end orchestration
"""

from .agent import SafetyAgent, DetectionResult, AgentDecision, AlertLevel
from .actions import AlertExecutor, TextWarning, AudioWarning, trigger_warning
from .pipeline import SafetyPipeline, PipelineResult
from .router import InVehicleCopilot, ModelRouter, DetectionResults, TTSOutput

__all__ = [
    # Main entry point
    "InVehicleCopilot",
    "ModelRouter",
    "DetectionResults",
    "TTSOutput",
    # Legacy components
    "SafetyAgent",
    "DetectionResult",
    "AgentDecision",
    "AlertLevel",
    "AlertExecutor",
    "TextWarning",
    "AudioWarning",
    "trigger_warning",
    "SafetyPipeline",
    "PipelineResult"
]
