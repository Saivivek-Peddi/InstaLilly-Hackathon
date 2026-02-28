# In-Vehicle AI Copilot: Multi-Modal Safety System

## Project Overview

A comprehensive on-device AI system that monitors:
- **External environment** (pedestrians, obstacles) via dashcam
- **Driver state** (drowsiness, distraction) via internal camera
- **Voice interaction** for commands and conversational AI

All running **on-device** with intelligent routing between 4 specialized LoRA-fine-tuned models.

---

## Team

| Member | Role | Models Responsible |
|--------|------|-------------------|
| **Person A** | ML Engineer | Pedestrian Detector, Voice Assistant |
| **Person B** | ML Engineer | Drowsiness Detector, Distraction Detector |

**Shared:** Agentic Router, API, UI, Deployment

---

## Infrastructure

### Development (Local)
- Code development, testing, API/UI
- Git push to GitHub

### Training (GCP VM)
```
IP:       34.123.38.202
User:     hackathon
VM Name:  hackathon-vm-hack-team13
SSH:      ssh hackathon@34.123.38.202
```

### VM Specs
```
┌─────────────────────────────────────────────────────────────┐
│  GPU:   NVIDIA RTX PRO 6000 (Blackwell) - 96GB VRAM         │
│  CUDA:  12.8                                                 │
│  RAM:   176 GB                                               │
│  CPU:   AMD EPYC 9B45 - 48 cores                            │
│  Disk:  170 GB available                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IN-VEHICLE AI COPILOT                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   EXTERNAL   │    │   INTERNAL   │    │    VOICE     │              │
│  │    CAMERA    │    │    CAMERA    │    │   (Gemma 3n) │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ Pedestrian   │    │ Drowsiness   │    │   Command    │              │
│  │ Detector     │    │ Detector     │    │   Parser     │              │
│  │(PaliGemma    │    │(PaliGemma    │    │ (Gemma 3n)   │              │
│  │  + LoRA)     │    │  + LoRA)     │    │              │              │
│  │  [Person A]  │    │  [Person B]  │    │  [Person A]  │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         │            ┌──────┴───────┐           │                       │
│         │            │ Distraction  │           │                       │
│         │            │ Detector     │           │                       │
│         │            │ [Person B]   │           │                       │
│         │            └──────┬───────┘           │                       │
│         │                   │                   │                       │
│         └───────────────────┼───────────────────┘                       │
│                             ▼                                           │
│                 ┌───────────────────────┐                               │
│                 │    CONTEXT ROUTER     │  [Shared]                     │
│                 │    (FunctionGemma)    │                               │
│                 └───────────┬───────────┘                               │
│                             ▼                                           │
│                 ┌───────────────────────┐                               │
│                 │   DECISION AGENT      │  [Shared]                     │
│                 │     (Gemma 3)         │                               │
│                 └───────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT WORKFLOW                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LOCAL (Mac)                      VM (96GB GPU)             │
│  ───────────                      ─────────────             │
│  • Write training code            • git pull                │
│  • Prepare datasets               • uv sync                 │
│  • Test with small samples        • Run fine-tuning         │
│  • Build API/UI                   • Save checkpoints        │
│  • git push                       • Upload to cloud storage │
│                                                             │
│       [Code] ──────GitHub──────> [Fine-tune]                │
│                                       │                     │
│       [Demo] <────Checkpoints─────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
InstaLilly-Hackathon/
├── models/
│   ├── pedestrian_detector/     # [Person A]
│   │   ├── data/
│   │   ├── notebooks/
│   │   ├── checkpoints/
│   │   └── metrics/
│   ├── drowsiness_detector/     # [Person B]
│   │   ├── data/
│   │   ├── notebooks/
│   │   ├── checkpoints/
│   │   └── metrics/
│   ├── distraction_detector/    # [Person B]
│   │   ├── data/
│   │   ├── notebooks/
│   │   ├── checkpoints/
│   │   └── metrics/
│   └── voice_assistant/         # [Person A]
│       ├── data/
│       ├── notebooks/
│       ├── checkpoints/
│       └── metrics/
├── agentic_system/              # [Shared]
├── api/                         # [Shared]
├── ui/                          # [Shared]
├── deployment/
├── presentation/
│   ├── metrics/
│   ├── screenshots/
│   └── demo_scripts/
├── scripts/                     # Setup & utility scripts
├── pyproject.toml              # UV dependencies
├── plan.md
└── README.md
```

---

## Phase 1: Setup (30 min)

### Local Setup
```bash
# Clone repo
git clone https://github.com/Saivivek-Peddi/InstaLilly-Hackathon.git
cd InstaLilly-Hackathon

# Install UV (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

### VM Setup (One-time)
```bash
# SSH into VM
ssh hackathon@34.123.38.202
# Password: 364d3f20

# Clone repo
git clone https://github.com/Saivivek-Peddi/InstaLilly-Hackathon.git
cd InstaLilly-Hackathon

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env

# Install dependencies
uv sync
```

---

## Phase 2: Data Collection (1 hr)

### Person A: Pedestrian + Voice Data

#### Pedestrian Detection
| Dataset | Size | Source |
|---------|------|--------|
| KITTI Pedestrian | 15,000 frames | http://www.cvlibs.net/datasets/kitti/ |
| CityPersons | 5,000 images | https://www.cityscapes-dataset.com/ |

#### Voice Commands
| Dataset | Size | Source |
|---------|------|--------|
| Car-Command | 8,500 commands | https://www.kaggle.com/datasets/oortdatahub/car-command |
| Google Speech Commands | 65,000 clips | TensorFlow datasets |

### Person B: Drowsiness + Distraction Data

#### Drowsiness Detection
| Dataset | Size | Source |
|---------|------|--------|
| Driver Drowsiness (DDD) | ~10K images | https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd |
| MRL Eye Dataset | Large-scale | https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset |
| FL3D | Frame-level | https://www.kaggle.com/datasets/matjazmuc/frame-level-driver-drowsiness-detection-fl3d |

#### Distraction Detection
| Dataset | Size | Source |
|---------|------|--------|
| State Farm | 22,000+ images | https://www.kaggle.com/c/state-farm-distracted-driver-detection |

---

## Phase 3: Baseline Benchmarking (1 hr)

**Both team members** run baseline evaluation on their models BEFORE fine-tuning.

### Metrics to Collect

```yaml
baseline_metrics:
  model_name: "PaliGemma-3B / Gemma-3n"
  task: "detection / classification"

  accuracy: float
  precision: float
  recall: float
  f1_score: float

  inference_time_ms: float
  model_size_mb: float

  confusion_matrix: saved as image
  sample_predictions: 10 examples
```

### Save Results
```bash
# Save to model's metrics folder
models/<model_name>/metrics/baseline.json
models/<model_name>/metrics/baseline_confusion.png
```

---

## Phase 4: LoRA Fine-tuning (3 hrs)

### Parallel Training Schedule

| Time | Person A (VM) | Person B (VM) |
|------|--------------|---------------|
| Hour 1 | Pedestrian Detector | Drowsiness Detector |
| Hour 2 | Voice Assistant | Distraction Detector |
| Hour 3 | Buffer / Debug | Buffer / Debug |

### LoRA Configuration (All Models)

```yaml
lora_config:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

training_args:
  learning_rate: 2e-4
  num_epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 4
  fp16: true
```

### Training Commands

```bash
# On VM - Person A
cd models/pedestrian_detector
uv run python train.py --config config.yaml

# On VM - Person B
cd models/drowsiness_detector
uv run python train.py --config config.yaml
```

### Save Checkpoints
```bash
models/<model_name>/checkpoints/
├── adapter_config.json
├── adapter_model.safetensors
└── training_args.json
```

---

## Phase 5: Post Fine-tuning Evaluation (30 min)

### Metrics to Collect

```yaml
finetuned_metrics:
  # Same as baseline +
  improvement:
    accuracy_delta: float
    accuracy_percent_gain: float

  lora_params: int
  trainable_percent: float
  training_time_hours: float
```

### Comparison Table (Fill During Hackathon)

| Model | Owner | Baseline | Fine-tuned | Improvement | Latency |
|-------|-------|----------|------------|-------------|---------|
| Pedestrian | Person A | __%  | __% | +__% | __ms |
| Drowsiness | Person B | __% | __% | +__% | __ms |
| Distraction | Person B | __% | __% | +__% | __ms |
| Voice | Person A | __% | __% | +__% | __ms |

---

## Phase 6: Integration (1 hr)

### Agentic Router [Shared Work]

```python
# agentic_system/router.py

class ContextRouter:
    PRIORITY = {
        "CRITICAL": 0,  # Collision imminent, driver asleep
        "HIGH": 1,      # Pedestrian nearby, drowsiness detected
        "MEDIUM": 2,    # Distraction detected
        "LOW": 3        # Voice command, conversation
    }

    def route(self, external_frame, internal_frame, audio):
        # Always run safety-critical
        pedestrian_result = self.pedestrian_model(external_frame)
        drowsiness_result = self.drowsiness_model(internal_frame)

        # Conditional routing
        if drowsiness_result.alert_level < "HIGH":
            distraction_result = self.distraction_model(internal_frame)

        if audio is not None:
            voice_result = self.voice_model(audio)

        return self.decision_agent.decide(results)
```

### API [Shared Work]

```python
# api/main.py
from fastapi import FastAPI, UploadFile

app = FastAPI(title="In-Vehicle AI Copilot")

@app.post("/analyze")
async def analyze(external: UploadFile, internal: UploadFile, audio: UploadFile = None):
    return router.route(external, internal, audio)
```

### UI [Shared Work]

```python
# ui/app.py
import gradio as gr

demo = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Image(source="webcam", label="Dashcam"),
        gr.Image(source="webcam", label="Driver Cam"),
        gr.Audio(source="microphone", label="Voice")
    ],
    outputs=[gr.JSON(), gr.Textbox(), gr.Audio()]
)
```

---

## Phase 7: Demo Preparation (30 min)

### Demo Checklist

- [ ] All 4 models loaded and working
- [ ] Router correctly prioritizing alerts
- [ ] UI showing both camera feeds
- [ ] Voice input/output working
- [ ] Metrics dashboard filled in
- [ ] Demo script rehearsed

### Demo Scenarios

1. **Normal driving** - System shows "All clear"
2. **Pedestrian detected** - Visual alert + bounding box
3. **Driver drowsy** - Audio warning + rest stop suggestion
4. **Driver distracted** - Warning + offer to help
5. **Voice command** - "Find rest stop" → Response

---

## Timeline Summary (8 Hours)

| Phase | Duration | Person A | Person B |
|-------|----------|----------|----------|
| Setup | 30 min | Local + VM setup | Local + VM setup |
| Data | 1 hr | Pedestrian + Voice data | Drowsiness + Distraction data |
| Baseline | 30 min | Benchmark models | Benchmark models |
| Fine-tune | 2.5 hr | Pedestrian → Voice | Drowsiness → Distraction |
| Evaluate | 30 min | Evaluate + metrics | Evaluate + metrics |
| Integrate | 1.5 hr | Router + API (together) | UI + Demo (together) |
| Demo Prep | 30 min | Polish + rehearse | Polish + rehearse |

---

## Git Workflow

### Before Starting Work
```bash
git pull origin main
```

### After Completing a Task
```bash
git add .
git commit -m "feat: <description>"
git push origin main
```

### Conflict Resolution
- Person A owns: `models/pedestrian_detector/`, `models/voice_assistant/`
- Person B owns: `models/drowsiness_detector/`, `models/distraction_detector/`
- Shared: `agentic_system/`, `api/`, `ui/` (coordinate before editing)

---

## Success Criteria

- [ ] All 4 models fine-tuned with measurable improvement over baseline
- [ ] Baseline vs Fine-tuned metrics documented for each model
- [ ] Agentic router with priority-based decisions working
- [ ] End-to-end demo with all 3 modalities (external cam, internal cam, voice)
- [ ] Inference latency < 100ms per model
- [ ] Live demo with webcam inputs
- [ ] Clear presentation with before/after metrics

---

## Presentation Metrics (Fill During Hackathon)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IN-VEHICLE AI COPILOT - FINAL RESULTS                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  MODELS TRAINED: 4          TOTAL INFERENCE: <___ms                    │
│  LoRA PARAMETERS: ~___M     AVG IMPROVEMENT: +___%                     │
│                                                                         │
│  ┌─────────────────┬────────────┬────────────┬─────────────┬─────────┐ │
│  │ Model           │ Baseline   │ Fine-tuned │ Improvement │ Latency │ │
│  ├─────────────────┼────────────┼────────────┼─────────────┼─────────┤ │
│  │ Pedestrian [A]  │    ___%    │    ___%    │    +___%    │   __ms  │ │
│  │ Drowsiness [B]  │    ___%    │    ___%    │    +___%    │   __ms  │ │
│  │ Distraction [B] │    ___%    │    ___%    │    +___%    │   __ms  │ │
│  │ Voice [A]       │    ___%    │    ___%    │    +___%    │   __ms  │ │
│  └─────────────────┴────────────┴────────────┴─────────────┴─────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Why This Project Wins

1. **Truly Multi-Modal**: Vision (2 cameras) + Audio - not just single modality
2. **Multi-Model Architecture**: 4 specialized models with intelligent routing
3. **Genuine On-Device Need**: Safety-critical latency, privacy, offline capability
4. **Agentic Behavior**: Not just detection, but contextual decision-making
5. **Real Impact**: Addresses 25% of accidents (distraction) + 20% fatal crashes (drowsiness)
6. **Clear Metrics**: Before/after comparison for every model
