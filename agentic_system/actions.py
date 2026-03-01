"""
Action Executors for Safety Warnings

Handles different types of alerts: text, audio, visual.
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Callable
from datetime import datetime


@dataclass
class Alert:
    """Alert to be displayed/played."""
    level: str  # none, low, medium, high, critical
    message: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TextWarning:
    """Text-based warning display."""

    COLORS = {
        "none": "\033[92m",      # Green
        "low": "\033[93m",       # Yellow
        "medium": "\033[93m",    # Yellow
        "high": "\033[91m",      # Red
        "critical": "\033[91m",  # Red
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    @classmethod
    def display(cls, alert: Alert):
        """Display text warning with colors."""
        color = cls.COLORS.get(alert.level, cls.RESET)
        icon = cls._get_icon(alert.level)

        print(f"\n{color}{cls.BOLD}")
        print("=" * 60)
        print(f"  {icon}  {alert.level.upper()} ALERT  {icon}")
        print("=" * 60)
        print(f"  {alert.message}")
        print("=" * 60)
        print(f"{cls.RESET}")

    @staticmethod
    def _get_icon(level: str) -> str:
        icons = {
            "none": "✓",
            "low": "ℹ",
            "medium": "⚠",
            "high": "⚠",
            "critical": "🚨"
        }
        return icons.get(level, "•")


class AudioWarning:
    """Audio-based warning (TTS)."""

    @staticmethod
    def play(message: str, blocking: bool = False):
        """
        Play audio warning using system TTS.

        Args:
            message: Text to speak
            blocking: Wait for audio to finish
        """
        if sys.platform == "darwin":  # macOS
            cmd = f'say "{message}"'
            if not blocking:
                cmd += " &"
            os.system(cmd)
        elif sys.platform == "win32":  # Windows
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(message)
                if blocking:
                    engine.runAndWait()
            except ImportError:
                print(f"[AUDIO] {message}")
        else:  # Linux
            try:
                os.system(f'espeak "{message}" &')
            except:
                print(f"[AUDIO] {message}")

    @staticmethod
    def beep(count: int = 1, frequency: int = 440):
        """Play alert beep sound."""
        if sys.platform == "darwin":
            for _ in range(count):
                os.system("afplay /System/Library/Sounds/Ping.aiff &")
                time.sleep(0.3)


class VisualWarning:
    """Visual warning effects."""

    @staticmethod
    def flash_terminal(times: int = 3, interval: float = 0.2):
        """Flash terminal background."""
        for _ in range(times):
            print("\033[41m" + " " * 60 + "\033[0m", end="\r")  # Red background
            time.sleep(interval)
            print(" " * 60, end="\r")
            time.sleep(interval)


class AlertExecutor:
    """
    Executes alerts through multiple channels.
    """

    def __init__(
        self,
        enable_text: bool = True,
        enable_audio: bool = False,
        enable_visual: bool = True,
        callback: Optional[Callable[[Alert], None]] = None
    ):
        self.enable_text = enable_text
        self.enable_audio = enable_audio
        self.enable_visual = enable_visual
        self.callback = callback
        self.history: list[Alert] = []

    def execute(self, level: str, message: str):
        """
        Execute alert through all enabled channels.

        Args:
            level: Alert level (none/low/medium/high/critical)
            message: Alert message
        """
        alert = Alert(level=level, message=message)
        self.history.append(alert)

        # Skip non-alerts
        if level == "none":
            if self.enable_text:
                print(f"\033[92m✓ {message}\033[0m")
            return

        # Text warning
        if self.enable_text:
            TextWarning.display(alert)

        # Visual flash for high/critical
        if self.enable_visual and level in ["high", "critical"]:
            VisualWarning.flash_terminal(times=2)

        # Audio warning
        if self.enable_audio:
            AudioWarning.play(message)

        # Custom callback
        if self.callback:
            self.callback(alert)

    def get_history(self) -> list[Alert]:
        """Get alert history."""
        return self.history

    def clear_history(self):
        """Clear alert history."""
        self.history = []


# Convenience function
def trigger_warning(level: str, message: str, audio: bool = False):
    """
    Trigger a warning with the given level and message.

    Args:
        level: Alert level
        message: Warning message
        audio: Enable audio output
    """
    executor = AlertExecutor(enable_audio=audio)
    executor.execute(level, message)


if __name__ == "__main__":
    print("Testing Alert System...")
    print("-" * 50)

    executor = AlertExecutor(enable_audio=False, enable_visual=True)

    # Test different levels
    print("\n1. Testing NONE level:")
    executor.execute("none", "Road clear. No hazards detected.")

    time.sleep(1)

    print("\n2. Testing HIGH level:")
    executor.execute("high", "CAUTION: Pedestrian detected ahead!")

    time.sleep(1)

    print("\n3. Testing CRITICAL level:")
    executor.execute("critical", "EMERGENCY: Multiple pedestrians in path!")

    print(f"\nAlert history: {len(executor.history)} alerts")
