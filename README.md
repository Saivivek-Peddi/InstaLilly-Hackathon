# In-Vehicle AI Copilot

> Multi-Modal On-Device Safety System for Vehicles

**Google DeepMind × InstaLILY On-Device AI Hackathon | Feb 2026**

---

## Overview

An intelligent in-vehicle AI system that monitors:
- **External environment** (pedestrians, obstacles) via dashcam
- **Driver state** (drowsiness, distraction) via internal camera
- **Voice commands** for hands-free interaction

All running **on-device** with intelligent routing between 4 specialized LoRA-fine-tuned Gemma models.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 IN-VEHICLE AI COPILOT                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   [Dashcam]          [Driver Cam]        [Microphone]       │
│       │                   │                   │             │
│       ▼                   ▼                   ▼             │
│  ┌─────────┐       ┌───────────┐       ┌──────────┐        │
│  │Pedestrian│       │Drowsiness │       │  Voice   │        │
│  │Detector │       │ Detector  │       │Assistant │        │
│  │PaliGemma│       │ PaliGemma │       │ Gemma 3n │        │
│  │ + LoRA  │       │  + LoRA   │       │          │        │
│  └────┬────┘       └─────┬─────┘       └────┬─────┘        │
│       │                  │                  │               │
│       │           ┌──────┴──────┐          │               │
│       │           │ Distraction │          │               │
│       │           │  Detector   │          │               │
│       │           │ Gemma 3n    │          │               │
│       │           └──────┬──────┘          │               │
│       │                  │                  │               │
│       └──────────────────┼──────────────────┘               │
│                          ▼                                  │
│                  ┌──────────────┐                           │
│                  │Context Router│                           │
│                  │ + Decision   │                           │
│                  │    Agent     │                           │
│                  └──────────────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Team

| Member | Models |
|--------|--------|
| Person A | Pedestrian Detector, Voice Assistant |
| Person B | Drowsiness Detector, Distraction Detector |

---

## Models

| Model | Base | Task | Dataset |
|-------|------|------|---------|
| Pedestrian Detector | PaliGemma 2 + LoRA | Object detection | KITTI, CityPersons |
| Drowsiness Detector | PaliGemma 2 + LoRA | State classification | DDD, MRL, FL3D |
| Distraction Detector | Gemma 3n + LoRA | Activity classification | State Farm |
| Voice Assistant | Gemma 3n | Command parsing | Car-Command |

---

## Quick Start

### Prerequisites

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

```bash
# Clone
git clone https://github.com/Saivivek-Peddi/InstaLilly-Hackathon.git
cd InstaLilly-Hackathon

# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

### Run Demo

```bash
# Start API
uv run uvicorn api.main:app --reload

# Start UI (new terminal)
uv run python ui/app.py
```

---

## Project Structure

```
├── models/
│   ├── pedestrian_detector/    # Person A
│   ├── drowsiness_detector/    # Person B
│   ├── distraction_detector/   # Person B
│   └── voice_assistant/        # Person A
├── agentic_system/             # Router + Decision Agent
├── api/                        # FastAPI backend
├── ui/                         # Gradio demo
├── presentation/               # Metrics & demo materials
├── scripts/                    # Utility scripts
└── plan.md                     # Detailed plan
```

---

## Results

| Model | Baseline | Fine-tuned | Improvement | Latency |
|-------|----------|------------|-------------|---------|
| Pedestrian | TBD | TBD | TBD | TBD |
| Drowsiness | TBD | TBD | TBD | TBD |
| Distraction | TBD | TBD | TBD | TBD |
| Voice | TBD | TBD | TBD | TBD |

---

## Why On-Device?

| Requirement | Cloud | On-Device |
|-------------|-------|-----------|
| **Latency** | 100-500ms | <50ms |
| **Privacy** | Data transmitted | Data stays local |
| **Offline** | Fails | Always works |
| **Cost** | $/inference | Free after deploy |

---

## Tech Stack

- **Models**: PaliGemma 2, Gemma 3n, FunctionGemma
- **Fine-tuning**: LoRA via PEFT
- **Package Manager**: UV
- **API**: FastAPI
- **UI**: Gradio

---

## Training Infrastructure

```
VM: hackathon-vm-hack-team13
GPU: NVIDIA RTX PRO 6000 (96GB VRAM)
RAM: 176 GB
CPU: AMD EPYC 9B45 (48 cores)
```

---

## License

MIT
