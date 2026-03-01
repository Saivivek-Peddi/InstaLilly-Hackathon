# In-Vehicle AI Copilot - Technical Documentation

## Overview

A multi-modal AI safety system for vehicles that processes camera inputs and generates real-time safety warnings using fine-tuned PaliGemma models with LoRA adapters.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        IN-VEHICLE AI COPILOT                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Internal Camera ──┬──→ Drowsiness Detector ──→ alert/drowsy/yawning  │
│   (Driver View)     └──→ Distraction Detector ──→ safe/texting/phone   │
│                                                         │               │
│   External Camera ────→ Pedestrian Detector ───→ yes/no │               │
│   (Road View)                                           │               │
│                                                         ▼               │
│                                              ┌──────────────────┐       │
│                                              │   Safety Agent   │       │
│                                              │  (Rule-based +   │       │
│                                              │   LLM reasoning) │       │
│                                              └────────┬─────────┘       │
│                                                       │                 │
│                                                       ▼                 │
│                                              ┌──────────────────┐       │
│                                              │   TTS Output     │       │
│                                              │  "CRITICAL:      │       │
│                                              │   Pedestrian!"   │       │
│                                              └──────────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Architecture

### Models

| Model | Base | Task | Classes | Accuracy |
|-------|------|------|---------|----------|
| Drowsiness Detector | PaliGemma2-3B | Driver state | alert, drowsy, yawning, eyes_closed | 92% |
| Distraction Detector | PaliGemma2-3B | Driver activity | safe_driving, texting_phone, talking_phone, other_activities, turning | 90% |
| Pedestrian Detector | PaliGemma2-3B | Road hazards | yes, no | 95%+ |

### Memory Optimization

**Problem**: Loading 3 separate PaliGemma models would require ~18GB+ VRAM.

**Solution**: Shared base model with LoRA adapter switching.

```python
# Single base model (4-bit quantized) ~3GB
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma2-3b-pt-448",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    device_map="auto"
)

# Switch adapters on demand (~37MB each)
peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
```

**Memory Usage**:
- Base model (4-bit): ~3GB VRAM
- Each LoRA adapter: ~37MB
- Total: ~3.5GB VRAM (vs ~18GB without optimization)

### LoRA Configuration

```python
LORA_CONFIG = {
    "r": 16,                    # Rank
    "lora_alpha": 32,           # Scaling factor
    "lora_dropout": 0.05,       # Dropout
    "target_modules": [         # Attention layers only
        "q_proj", "k_proj",
        "v_proj", "o_proj"
    ],
    "task_type": "CAUSAL_LM"
}
```

### Training Details

**Key Fix**: Using `suffix=` parameter in PaliGemmaProcessor for proper label creation.

```python
# WRONG - causes training collapse (0% accuracy)
inputs = processor(
    text=f"Driver state?\n{label}",  # Label in text
    images=image,
    return_tensors="pt"
)

# CORRECT - proper label masking (92% accuracy)
inputs = processor(
    text="Driver state?\n",
    images=image,
    suffix=label,  # Label as suffix - creates proper labels!
    return_tensors="pt",
    padding="max_length",
    max_length=1200
)
```

**Training Parameters**:
- Samples: 1000 (balanced, ~250 per class)
- Epochs: 2-3
- Batch size: 4
- Learning rate: 2e-4
- Checkpoints: Every 25 steps

## File Structure

```
hackathon-installilly/
├── agentic_system/
│   ├── __init__.py          # Exports
│   ├── router.py            # Main router (JSON → Models → TTS)
│   ├── agent.py             # Ollama-based reasoning
│   ├── actions.py           # Alert executors
│   └── pipeline.py          # Legacy pipeline
├── api/
│   ├── copilot_server.py    # FastAPI JSON endpoint
│   └── detection_server.py  # Legacy detection server
├── models/
│   ├── drowsiness_detector/
│   │   ├── train.py         # Training script
│   │   ├── prepare_data.py  # Data preparation
│   │   └── checkpoints/final/  # Fine-tuned LoRA
│   ├── distraction_detector/
│   │   ├── train.py
│   │   ├── prepare_data.py
│   │   └── checkpoints/final/
│   ├── pedestrian_detector/
│   │   ├── train.py
│   │   └── checkpoints/final/
│   └── voice_assistant/
│       └── assistant.py     # Function calling (text-based)
├── demo_full.py             # CLI demo
├── test_api.py              # API test script
├── test_images/             # Local test images
└── DOCUMENT.md              # This file
```

## API Reference

### Endpoints

#### `GET /health`
Health check endpoint.

**Response**:
```json
{
    "status": "healthy",
    "gpu_available": true,
    "gpu_name": "NVIDIA RTX PRO 6000"
}
```

#### `POST /process`
Process camera inputs and return safety warning.

**Request**:
```json
{
    "internal_camera": "/path/to/driver.jpg",
    "external_camera": "/path/to/road.jpg"
}
```

Or with base64:
```json
{
    "internal_camera": "data:image/jpeg;base64,/9j/4AAQ...",
    "external_camera": "data:image/png;base64,iVBORw0..."
}
```

**Response**:
```json
{
    "detections": {
        "driver_state": "drowsy",
        "driver_activity": "texting_phone",
        "pedestrian_detected": true,
        "drowsiness_time_ms": 245.3,
        "distraction_time_ms": 198.7,
        "pedestrian_time_ms": 156.2
    },
    "decision": {
        "alert_level": "critical",
        "message": "CRITICAL: You appear drowsy and Stop texting and Pedestrian ahead!",
        "reasoning": "Detected: You appear drowsy, Stop texting while driving, Pedestrian ahead"
    },
    "audio_file": "/tmp/warning.wav",
    "total_time_ms": 612.4
}
```

### Alert Levels

| Level | Conditions | Example |
|-------|------------|---------|
| `none` | All clear | Safe driving, no pedestrians |
| `low` | Minor warning | - |
| `medium` | Yawning, talking on phone | "Warning: You are yawning!" |
| `high` | Single serious issue | Pedestrian OR drowsy |
| `critical` | Multiple issues | Drowsy + Pedestrian |

## Setup & Running

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- UV package manager

### VM Setup

```bash
# SSH to VM
ssh hackathon@34.123.38.202

# Clone/update repo
cd ~/InstaLilly-Hackathon
git pull

# Install dependencies
uv sync --extra ml

# Verify checkpoints exist
ls models/*/checkpoints/final/
```

### Running the API

**On VM**:
```bash
cd ~/InstaLilly-Hackathon
uv run uvicorn api.copilot_server:app --host 0.0.0.0 --port 8000
```

**Port forwarding (local)**:
```bash
ssh -L 8000:localhost:8000 hackathon@34.123.38.202
```

### Running the Demo

**CLI Demo**:
```bash
python demo_full.py --internal driver.jpg --external road.jpg
```

**API Test**:
```bash
python test_api.py
```

## Performance

### Inference Latency

| Operation | Time |
|-----------|------|
| Model loading (first call) | ~6-8 seconds |
| Adapter switching | ~200ms |
| Single detection | ~100-300ms |
| Full pipeline (3 detections) | ~600-900ms |

### GPU Memory

| Configuration | VRAM Usage |
|---------------|------------|
| Single model (fp16) | ~6GB |
| Single model (4-bit) | ~3GB |
| 3 models naive | ~18GB |
| **Shared base + LoRA** | **~3.5GB** |

## Training Results

### Drowsiness Detector

| Metric | Baseline | Fine-tuned |
|--------|----------|------------|
| Accuracy | 3% | **92%** |
| Training samples | - | 1000 |
| Epochs | - | 3 |

### Distraction Detector

| Metric | Baseline | Fine-tuned |
|--------|----------|------------|
| Accuracy | 10% | **90%** |
| Training samples | - | 1000 |
| Epochs | - | 2 |

### Pedestrian Detector

| Metric | Baseline | Fine-tuned |
|--------|----------|------------|
| Accuracy | ~60% | **95%+** |
| Training samples | - | ~1500 |
| Epochs | - | 2 |

## Key Optimizations

### 1. Shared Base Model
Instead of loading 3 separate 3B models, we load one quantized base and swap LoRA adapters.

### 2. 4-bit Quantization
Using BitsAndBytes NF4 quantization reduces model size by 4x with minimal accuracy loss.

### 3. Attention-Only LoRA
Only training attention layers (q, k, v, o projections) reduces trainable parameters while maintaining performance.

### 4. Suffix-based Label Creation
Using the `suffix=` parameter in PaliGemmaProcessor ensures proper label masking during training.

### 5. Balanced Sampling
Training on balanced datasets (equal samples per class) prevents model bias.

## Troubleshooting

### Training Collapse (0% accuracy, loss stuck at ~12.48)
**Cause**: Using text concatenation instead of `suffix=` parameter.
**Fix**: Use `processor(text=prompt, images=image, suffix=label, ...)`.

### Out of Memory
**Cause**: Loading multiple models or large batch sizes.
**Fix**: Use 4-bit quantization, shared base model, batch_size=4.

### Slow First Inference
**Cause**: Model loading on first call.
**Fix**: Pre-warm the model with a dummy request.

### Port Already in Use
```bash
pkill -f uvicorn
# Then restart
```

## Future Improvements

1. **Voice Input**: Add Whisper or Gemma 3n for speech-to-text
2. **Streaming**: WebSocket support for real-time video processing
3. **Edge Deployment**: ONNX/TensorRT optimization for embedded devices
4. **Multi-GPU**: Parallel inference across multiple GPUs
5. **Caching**: Cache adapter weights in memory for faster switching

## Contributors

- **Person A**: Pedestrian Detector, Voice Assistant
- **Person B**: Drowsiness Detector, Distraction Detector
- **Shared**: Agentic System, API, Integration

## License

Hackathon project for Google DeepMind × InstaLILY On-Device AI Hackathon.
