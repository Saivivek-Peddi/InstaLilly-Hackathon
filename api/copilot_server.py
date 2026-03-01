#!/usr/bin/env python3
"""
In-Vehicle AI Copilot API Server

Simple JSON API:
- POST /process: Process camera inputs and get TTS warning

Usage:
    uvicorn api.copilot_server:app --host 0.0.0.0 --port 8000

Example request:
    curl -X POST http://localhost:8000/process \
        -H "Content-Type: application/json" \
        -d '{"internal_camera": "/path/to/driver.jpg", "external_camera": "/path/to/road.jpg"}'
"""

import sys
import base64
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add project to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from agentic_system.router import InVehicleCopilot


# Pydantic models
class CameraInput(BaseModel):
    """Input model for camera images."""
    internal_camera: Optional[str] = None  # Path or base64
    external_camera: Optional[str] = None  # Path or base64


class DetectionOutput(BaseModel):
    """Detection results."""
    driver_state: str
    driver_activity: str
    pedestrian_detected: bool
    drowsiness_time_ms: float
    distraction_time_ms: float
    pedestrian_time_ms: float


class DecisionOutput(BaseModel):
    """Agent decision."""
    alert_level: str
    message: str
    reasoning: str


class ProcessResponse(BaseModel):
    """Full response."""
    detections: DetectionOutput
    decision: DecisionOutput
    audio_file: str
    total_time_ms: float


# Initialize app
app = FastAPI(
    title="In-Vehicle AI Copilot",
    description="Multi-modal safety system with TTS output",
    version="1.0.0"
)

# Global copilot instance (lazy loaded)
copilot = None


def get_copilot():
    """Get or create copilot instance."""
    global copilot
    if copilot is None:
        print("Initializing copilot...")
        copilot = InVehicleCopilot()
    return copilot


def decode_base64_image(data: str, prefix: str = "image") -> str:
    """Decode base64 image and save to temp file."""
    # Remove data URL prefix if present
    if "," in data:
        data = data.split(",")[1]

    image_bytes = base64.b64decode(data)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", prefix=prefix, delete=False) as f:
        f.write(image_bytes)
        return f.name


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "service": "In-Vehicle AI Copilot",
        "version": "1.0.0",
        "endpoints": {
            "/process": "POST - Process camera inputs",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
def health():
    """Health check."""
    import torch
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@app.post("/process", response_model=ProcessResponse)
def process(input_data: CameraInput):
    """
    Process camera inputs and return warning.

    Input can be:
    - File paths: {"internal_camera": "/path/to/image.jpg"}
    - Base64: {"internal_camera": "data:image/jpeg;base64,..."}
    """
    try:
        cop = get_copilot()

        # Prepare input JSON
        input_json = {}

        # Handle internal camera
        if input_data.internal_camera:
            if input_data.internal_camera.startswith("data:") or len(input_data.internal_camera) > 500:
                # Base64 encoded
                input_json["internal_camera"] = decode_base64_image(
                    input_data.internal_camera, "internal_"
                )
            else:
                # File path
                input_json["internal_camera"] = input_data.internal_camera

        # Handle external camera
        if input_data.external_camera:
            if input_data.external_camera.startswith("data:") or len(input_data.external_camera) > 500:
                # Base64 encoded
                input_json["external_camera"] = decode_base64_image(
                    input_data.external_camera, "external_"
                )
            else:
                # File path
                input_json["external_camera"] = input_data.external_camera

        # Process
        result = cop.process(input_json, speak=True)

        return ProcessResponse(
            detections=DetectionOutput(**result["detections"]),
            decision=DecisionOutput(**result["decision"]),
            audio_file=result["audio_file"],
            total_time_ms=result["total_time_ms"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{filename}")
def get_audio(filename: str):
    """Get generated audio file."""
    audio_path = Path(f"/tmp/{filename}")
    if audio_path.exists():
        return FileResponse(audio_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio file not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
