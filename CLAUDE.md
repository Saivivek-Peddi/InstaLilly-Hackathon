# CLAUDE.md - Project Context for Claude Code

## Project Overview

This is an **In-Vehicle AI Copilot** project for the Google DeepMind × InstaLILY On-Device AI Hackathon.

We're building a multi-modal safety system that runs on-device with 4 specialized LoRA-fine-tuned Gemma models.

## Architecture

```
External Camera → Pedestrian Detector (PaliGemma + LoRA)
Internal Camera → Drowsiness Detector (PaliGemma + LoRA)
                → Distraction Detector (Gemma 3n + LoRA)
Microphone     → Voice Assistant (Gemma 3n)
                         ↓
              Context Router (FunctionGemma)
                         ↓
              Decision Agent (Gemma 3)
```

## Team & Ownership

| Person | Models | Folders |
|--------|--------|---------|
| Person A | Pedestrian, Voice | `models/pedestrian_detector/`, `models/voice_assistant/` |
| Person B | Drowsiness, Distraction | `models/drowsiness_detector/`, `models/distraction_detector/` |
| Shared | Router, API, UI | `agentic_system/`, `api/`, `ui/` |

## Key Files

- `plan.md` - Detailed implementation plan with timeline
- `presentation/metrics/` - Metrics templates to fill during hackathon
- `presentation/demo_scripts/demo_flow.md` - Demo script for presentation

## Development Workflow

1. **Local**: Write code, test with small samples
2. **Push**: `git push` to GitHub
3. **VM**: Pull and run fine-tuning on GPU VM
4. **Checkpoints**: Upload trained models to cloud storage

## VM Access (for fine-tuning)

```
IP: 34.123.38.202
User: hackathon
GPU: NVIDIA RTX PRO 6000 (96GB VRAM)
```

## Commands

```bash
# Local setup
uv sync

# Run API
uv run uvicorn api.main:app --reload

# Run UI
uv run python ui/app.py

# On VM - training
uv run python models/<model>/train.py
```

## Datasets

| Model | Dataset | Source |
|-------|---------|--------|
| Pedestrian | KITTI, CityPersons | KITTI website, Cityscapes |
| Drowsiness | DDD, MRL, FL3D | Kaggle |
| Distraction | State Farm | Kaggle |
| Voice | Car-Command | Kaggle |

## Important Notes

- Always collect **baseline metrics BEFORE fine-tuning**
- Save metrics to `models/<model>/metrics/` folder
- Use LoRA config: r=16, alpha=32, dropout=0.05
- Target inference latency: <50ms per model
- Fill presentation metrics dashboard during hackathon

## Tech Stack

- Package manager: UV
- ML: PyTorch, Transformers, PEFT
- API: FastAPI
- UI: Gradio
- Models: PaliGemma 2, Gemma 3n, FunctionGemma
