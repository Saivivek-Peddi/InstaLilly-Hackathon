"""
Agentic System for In-Vehicle AI Copilot

Components:
- agent.py: Ollama-based decision agent
- actions.py: Alert/warning executors
- pipeline.py: End-to-end orchestration
"""

from .agent import SafetyAgent, DetectionResult, AgentDecision, AlertLevel
from .actions import AlertExecutor, TextWarning, AudioWarning, trigger_warning
from .pipeline import SafetyPipeline, PipelineResult

__all__ = [
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
