#!/usr/bin/env python3
"""
Inference Script for Pedestrian Detection Model

Run the fine-tuned PaliGemma model on an image to detect pedestrians.

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --image path/to/image.jpg --visualize

Example:
    python inference.py --image data/processed/images/penn_fudan_pedestrian_000121.jpg
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

try:
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    from peft import PeftModel
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML dependencies not available. Install with: pip install transformers peft torch")

# Paths
MODEL_DIR = Path(__file__).parent
CHECKPOINT_DIR = MODEL_DIR / "checkpoints" / "final"
BASE_MODEL = "google/paligemma2-3b-pt-448"

# Detection prompt (same as training)
DETECTION_PROMPT = """Analyze this dashcam image for pedestrian detection.

Task: Identify if there are any pedestrians in the scene and assess the safety situation.

Categories:
- no_pedestrian: No pedestrians visible in the frame
- pedestrian: One pedestrian detected
- multiple_pedestrians: Multiple pedestrians detected

Provide the detection result and safety assessment."""


def load_model(use_finetuned=True):
    """Load the pedestrian detection model."""
    print(f"🔧 Loading model...")

    # Load processor
    if use_finetuned and CHECKPOINT_DIR.exists():
        processor = PaliGemmaProcessor.from_pretrained(str(CHECKPOINT_DIR))
        print(f"   ✓ Loaded processor from checkpoint")
    else:
        processor = PaliGemmaProcessor.from_pretrained(BASE_MODEL)
        print(f"   ✓ Loaded processor from {BASE_MODEL}")

    # Load base model
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print(f"   ✓ Loaded base model: {BASE_MODEL}")

    # Load LoRA adapter if using finetuned
    if use_finetuned and CHECKPOINT_DIR.exists():
        model = PeftModel.from_pretrained(model, str(CHECKPOINT_DIR))
        print(f"   ✓ Loaded LoRA adapter from checkpoint")
    else:
        print(f"   ⚠ Using base model (no finetuning)")

    model.eval()
    return model, processor


def detect_pedestrians(model, processor, image_path, device="cuda"):
    """Run pedestrian detection on an image."""
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Process inputs
    inputs = processor(
        text=DETECTION_PROMPT,
        images=image,
        return_tensors="pt"
    ).to(device)

    # Generate response
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )
    inference_time = (time.time() - start_time) * 1000  # ms

    # Decode response
    response = processor.decode(outputs[0], skip_special_tokens=True)

    # Parse detection result
    response_lower = response.lower()
    if "multiple_pedestrian" in response_lower or "multiple pedestrian" in response_lower:
        detection = "multiple_pedestrians"
        safety = "⚠️  HIGH ALERT"
    elif "no_pedestrian" in response_lower or "no pedestrian" in response_lower:
        detection = "no_pedestrian"
        safety = "✅ CLEAR"
    elif "pedestrian" in response_lower:
        detection = "pedestrian"
        safety = "⚡ CAUTION"
    else:
        detection = "unknown"
        safety = "❓ UNKNOWN"

    return {
        "detection": detection,
        "safety_status": safety,
        "response": response,
        "inference_time_ms": inference_time,
        "image_path": str(image_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Pedestrian Detection Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--base_only", action="store_true", help="Use base model without finetuning")
    parser.add_argument("--visualize", action="store_true", help="Display image with result")
    args = parser.parse_args()

    if not ML_AVAILABLE:
        print("❌ ML dependencies not available")
        return

    # Check image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return

    # Load model
    model, processor = load_model(use_finetuned=not args.base_only)

    # Run detection
    print(f"\n🔍 Analyzing image: {image_path.name}")
    print("-" * 50)

    result = detect_pedestrians(model, processor, image_path)

    # Display results
    print(f"\n{'='*50}")
    print(f"🚶 PEDESTRIAN DETECTION RESULT")
    print(f"{'='*50}")
    print(f"Detection:     {result['detection']}")
    print(f"Safety Status: {result['safety_status']}")
    print(f"Inference:     {result['inference_time_ms']:.1f} ms")
    print(f"{'='*50}")
    print(f"\nFull Response:\n{result['response']}")

    # Visualize if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            img = Image.open(image_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(f"Detection: {result['detection']} | {result['safety_status']}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("\n⚠️  matplotlib not available for visualization")


if __name__ == "__main__":
    main()
