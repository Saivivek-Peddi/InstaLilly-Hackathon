"""
FastAPI Detection Server for Pedestrian Detection

Serves the fine-tuned PaliGemma model for inference.
Run on VM with GPU.

Usage:
    uvicorn api.detection_server:app --host 0.0.0.0 --port 8000
"""

import io
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ML imports (will fail on Mac without GPU - that's OK)
try:
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    from peft import PeftModel
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


# Paths
MODEL_DIR = Path(__file__).parent.parent / "models" / "pedestrian_detector"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints" / "final"
BASE_MODEL = "google/paligemma2-3b-pt-448"

# Global model variables
model = None
processor = None


class DetectionResponse(BaseModel):
    """Response from detection endpoint."""
    pedestrian_detected: bool
    confidence: float
    raw_response: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None


def load_model():
    """Load the fine-tuned PaliGemma model."""
    global model, processor

    if not ML_AVAILABLE:
        print("ML dependencies not available")
        return False

    print(f"Loading model from {CHECKPOINT_DIR}...")

    try:
        # Load processor
        if CHECKPOINT_DIR.exists():
            processor = PaliGemmaProcessor.from_pretrained(str(CHECKPOINT_DIR))
        else:
            processor = PaliGemmaProcessor.from_pretrained(BASE_MODEL)

        # Load base model
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Load LoRA adapter
        if CHECKPOINT_DIR.exists():
            model = PeftModel.from_pretrained(model, str(CHECKPOINT_DIR))
            print(f"Loaded LoRA adapter from {CHECKPOINT_DIR}")
        else:
            print("Warning: No checkpoint found, using base model")

        model.eval()
        print("Model loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield
    # Cleanup on shutdown
    global model, processor
    model = None
    processor = None


# Create FastAPI app
app = FastAPI(
    title="Pedestrian Detection API",
    description="PaliGemma-based pedestrian detection for in-vehicle safety",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available() if ML_AVAILABLE else False
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        gpu_available=gpu_available,
        gpu_name=gpu_name
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect_pedestrian(image: UploadFile = File(...)):
    """
    Detect pedestrians in an uploaded image.

    Args:
        image: Image file (JPEG, PNG)

    Returns:
        DetectionResponse with detection result
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read image
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Run inference
    start_time = time.time()

    prompt = "Is there a pedestrian?"
    inputs = processor(text=prompt, images=img, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )

    response = processor.decode(outputs[0], skip_special_tokens=True)
    inference_time = (time.time() - start_time) * 1000

    # Parse response
    answer = response.split("?")[-1].strip().lower()
    detected = "yes" in answer or answer.startswith("y")

    return DetectionResponse(
        pedestrian_detected=detected,
        confidence=1.0 if detected else 0.0,
        raw_response=answer[:50],
        inference_time_ms=inference_time
    )


@app.post("/detect/batch")
async def detect_batch(images: list[UploadFile] = File(...)):
    """
    Detect pedestrians in multiple images.

    Args:
        images: List of image files

    Returns:
        List of DetectionResponse
    """
    results = []
    for img in images:
        result = await detect_pedestrian(img)
        results.append(result)
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
