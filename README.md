# 🔫 SHOTGUN

## A Mixture-of-Experts Agentic Vehicle Copilot

> Multi-Modal On-Device Safety System with Shared Backbone + LoRA Experts

**Google DeepMind × InstaLILY On-Device AI Hackathon | Feb 2026**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

An intelligent in-vehicle AI copilot that monitors driver safety in real-time:

| Input | Model | Detection |
|-------|-------|-----------|
| 📷 Internal Camera | Drowsiness Expert | alert, drowsy, yawning, eyes_closed |
| 📷 Internal Camera | Distraction Expert | safe_driving, texting, talking_phone |
| 📷 External Camera | Pedestrian Expert | yes/no |
| 🎤 Microphone | Voice Assistant | Function calling |

**Key Features:**
- ✅ **On-Device** - No cloud, no latency, full privacy
- ✅ **Memory Efficient** - 3.1GB VRAM (8x reduction via shared backbone)
- ✅ **Fast** - <1 second end-to-end inference
- ✅ **Accurate** - 90%+ accuracy across all tasks

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SHOTGUN                                        │
│                   Mixture-of-Experts Vehicle Copilot                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌─────────────────────┐                             │
│                         │   PaliGemma2-3B     │                             │
│                         │  (Shared Backbone)  │                             │
│                         │   4-bit Quantized   │                             │
│                         │       ~3 GB         │                             │
│                         └──────────┬──────────┘                             │
│                                    │                                        │
│              ┌─────────────────────┼─────────────────────┐                  │
│              │                     │                     │                  │
│              ▼                     ▼                     ▼                  │
│       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐          │
│       │    LoRA     │       │    LoRA     │       │    LoRA     │          │
│       │ Drowsiness  │       │ Distraction │       │ Pedestrian  │          │
│       │   Expert    │       │   Expert    │       │   Expert    │          │
│       │   ~37 MB    │       │   ~37 MB    │       │   ~37 MB    │          │
│       └──────┬──────┘       └──────┬──────┘       └──────┬──────┘          │
│              │                     │                     │                  │
│              └─────────────────────┼─────────────────────┘                  │
│                                    ▼                                        │
│                         ┌─────────────────────┐                             │
│                         │    Safety Agent     │                             │
│                         │  (Decision Engine)  │                             │
│                         └──────────┬──────────┘                             │
│                                    ▼                                        │
│                         ┌─────────────────────┐                             │
│                         │     TTS Output      │                             │
│                         │  "CRITICAL: Wake    │                             │
│                         │   up! Pedestrian!"  │                             │
│                         └─────────────────────┘                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Results

| Model | Baseline | Fine-tuned | Improvement | Samples |
|-------|----------|------------|-------------|---------|
| **Drowsiness** | 3% | **92%** | +89% | 1,000 |
| **Distraction** | 10% | **90%** | +80% | 1,000 |
| **Pedestrian** | 60% | **95%+** | +35% | 1,500 |

### Memory Efficiency

| Approach | VRAM | Reduction |
|----------|------|-----------|
| 3 separate models (fp16) | ~24 GB | - |
| 3 separate models (4-bit) | ~9 GB | 2.7x |
| **Shared backbone + LoRA** | **~3.1 GB** | **8x** |

### Inference Latency

| Operation | Time |
|-----------|------|
| Single detection | 100-300ms |
| Full pipeline (3 models) | ~700ms |
| First call (model loading) | ~6-8s |

---

## Quick Start

### Prerequisites

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
# Clone repository
git clone https://github.com/Saivivek-Peddi/InstaLilly-Hackathon.git
cd InstaLilly-Hackathon

# Install dependencies
uv sync --extra ml
```

### Run Demo (CLI)

```bash
# Process images
uv run python demo_full.py --internal driver.jpg --external road.jpg
```

### Run API Server

```bash
# Start server
uv run uvicorn api.copilot_server:app --host 0.0.0.0 --port 8000

# Test health
curl http://localhost:8000/health

# Process images
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"internal_camera": "/path/to/driver.jpg", "external_camera": "/path/to/road.jpg"}'
```

### Run Tests

```bash
# API test (requires server running)
uv run python test_api.py
```

---

## Project Structure

```
├── agentic_system/
│   ├── router.py              # Main router (JSON → Models → TTS)
│   ├── agent.py               # Ollama-based reasoning
│   ├── actions.py             # Alert executors
│   └── pipeline.py            # Legacy pipeline
├── api/
│   ├── copilot_server.py      # FastAPI JSON endpoint
│   └── detection_server.py    # Legacy detection server
├── models/
│   ├── drowsiness_detector/
│   │   ├── train.py           # Training script
│   │   ├── prepare_data.py    # Data preparation
│   │   └── checkpoints/final/ # Fine-tuned LoRA
│   ├── distraction_detector/
│   │   ├── train.py
│   │   ├── prepare_data.py
│   │   └── checkpoints/final/
│   ├── pedestrian_detector/
│   │   ├── train.py
│   │   └── checkpoints/final/
│   └── voice_assistant/
│       └── assistant.py       # Function calling
├── demo_full.py               # CLI demo
├── test_api.py                # API test script
├── DOCUMENT.md                # Technical documentation
├── PRESENTATION.md            # Presentation slides
└── README.md                  # This file
```

---

## API Reference

### `POST /process`

Process camera inputs and return safety warning.

**Request:**
```json
{
  "internal_camera": "/path/to/driver.jpg",
  "external_camera": "/path/to/road.jpg"
}
```

**Response:**
```json
{
  "detections": {
    "driver_state": "drowsy",
    "driver_activity": "texting_phone",
    "pedestrian_detected": true
  },
  "decision": {
    "alert_level": "critical",
    "message": "CRITICAL: You appear drowsy and Stop texting and Pedestrian ahead!",
    "reasoning": "Multiple hazards detected"
  },
  "audio_file": "/tmp/warning.wav",
  "total_time_ms": 712.4
}
```

### Alert Levels

| Level | Condition | Example |
|-------|-----------|---------|
| `none` | All clear | Safe driving, no pedestrians |
| `medium` | Minor issue | Yawning, talking on phone |
| `high` | Single serious issue | Pedestrian OR drowsy |
| `critical` | Multiple issues | Drowsy + Pedestrian |

---

## Training

### LoRA Configuration

```python
LORA_CONFIG = {
    "r": 16,                           # Rank
    "lora_alpha": 32,                  # Scaling factor
    "lora_dropout": 0.05,              # Dropout
    "target_modules": [                # Attention only
        "q_proj", "k_proj",
        "v_proj", "o_proj"
    ]
}
```

### Key Training Fix

```python
# WRONG - causes 0% accuracy
inputs = processor(text=f"Driver state?\n{label}", images=image)

# CORRECT - use suffix parameter
inputs = processor(text="Driver state?\n", images=image, suffix=label)
```

### Run Training

```bash
# On GPU VM
cd models/drowsiness_detector
uv run python train.py --epochs 3 --batch_size 4

cd models/distraction_detector
uv run python train.py --epochs 2 --batch_size 4
```

---

## Why On-Device?

| Requirement | Cloud | On-Device (SHOTGUN) |
|-------------|-------|---------------------|
| **Latency** | 100-500ms | <100ms |
| **Privacy** | Data transmitted | Data stays local |
| **Offline** | Fails | Always works |
| **Cost** | $/inference | Free after deploy |
| **Security** | Attack surface | No network exposure |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Base Model | PaliGemma2-3B-pt-448 |
| Fine-tuning | LoRA (PEFT) |
| Quantization | BitsAndBytes NF4 |
| Package Manager | UV |
| API | FastAPI |
| TTS | espeak-ng |

---

## Hardware Requirements

### Training
- GPU: 24GB+ VRAM (or 4-bit quantization)
- RAM: 32GB+
- Storage: 50GB+

### Inference
- GPU: 4GB+ VRAM (with 4-bit quantization)
- RAM: 8GB+
- Storage: 5GB

### Our Setup
```
VM: hackathon-vm-hack-team13
GPU: NVIDIA RTX PRO 6000 (96GB VRAM)
RAM: 176 GB
CPU: AMD EPYC 9B45 (48 cores)
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [DOCUMENT.md](DOCUMENT.md) | Technical documentation |
| [PRESENTATION.md](PRESENTATION.md) | Hackathon presentation |
| [CLAUDE.md](CLAUDE.md) | Project context for Claude |

---

## Team

| Member | Responsibility |
|--------|----------------|
| Person A | Pedestrian Detector, Voice Assistant |
| Person B | Drowsiness Detector, Distraction Detector |
| Shared | Agentic System, API, Integration |

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Google DeepMind for PaliGemma2 and Gemma models
- InstaLILY for hosting the hackathon
- Kaggle for datasets (State Farm, DDD, MRL)

---

<p align="center">
  <b>🔫 SHOTGUN</b><br>
  <i>Making roads safer, one inference at a time.</i>
</p>
