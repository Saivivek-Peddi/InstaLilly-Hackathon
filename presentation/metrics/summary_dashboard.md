# In-Vehicle AI Copilot - Results Dashboard

## Project Summary

| Attribute | Value |
|-----------|-------|
| **Team** | [Your Team Name] |
| **Hackathon** | Google DeepMind × InstaLILY On-Device AI Hackathon |
| **Date** | February 28, 2026 |
| **Total Models** | 4 |
| **Total Training Time** | ~4 hours |

---

## Overall Results

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IN-VEHICLE AI COPILOT - FINAL RESULTS                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ╔═══════════════════════════════════════════════════════════════════╗ │
│  ║  MODELS TRAINED: 4          TOTAL INFERENCE: <100ms               ║ │
│  ║  LoRA PARAMETERS: ~50M      IMPROVEMENT: +XX% avg                 ║ │
│  ╚═══════════════════════════════════════════════════════════════════╝ │
│                                                                         │
│  ┌─────────────────┬────────────┬────────────┬─────────────┬─────────┐ │
│  │ Model           │ Baseline   │ Fine-tuned │ Improvement │ Latency │ │
│  ├─────────────────┼────────────┼────────────┼─────────────┼─────────┤ │
│  │ Pedestrian Det. │    ___%    │    ___%    │    +___%    │   __ms  │ │
│  │ Drowsiness Det. │    ___%    │    ___%    │    +___%    │   __ms  │ │
│  │ Distraction Det.│    ___%    │    ___%    │    +___%    │   __ms  │ │
│  │ Voice Assistant │    ___%    │    ___%    │    +___%    │   __ms  │ │
│  └─────────────────┴────────────┴────────────┴─────────────┴─────────┘ │
│                                                                         │
│  TOTAL SYSTEM LATENCY: ___ms (Target: <100ms) ✓/✗                      │
│  MEMORY FOOTPRINT: ___MB (Edge-deployable: <8GB) ✓/✗                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Model-by-Model Results

### 1. Pedestrian Detector (PaliGemma + LoRA)

| Metric | Baseline | Fine-tuned | Change |
|--------|----------|------------|--------|
| mAP@0.5 | | | |
| mAP@0.5:0.95 | | | |
| Precision | | | |
| Recall | | | |
| Inference (ms) | | | |

**Training Details:**
- Dataset: KITTI + CityPersons
- Training samples: ___
- LoRA rank: 16
- Training time: ___

### 2. Drowsiness Detector (PaliGemma + LoRA)

| Metric | Baseline | Fine-tuned | Change |
|--------|----------|------------|--------|
| Accuracy | | | |
| Precision | | | |
| Recall | | | |
| F1-Score | | | |
| Inference (ms) | | | |

**Per-Class Results:**
| State | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Alert | | | |
| Drowsy | | | |
| Yawning | | | |
| Eyes Closed | | | |

### 3. Distraction Detector (Gemma 3n + LoRA)

| Metric | Baseline | Fine-tuned | Change |
|--------|----------|------------|--------|
| Accuracy | | | |
| Top-3 Accuracy | | | |
| Inference (ms) | | | |

**Per-Class Results:**
| Activity | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Safe Driving | | | |
| Texting (R) | | | |
| Texting (L) | | | |
| Phone (R) | | | |
| Phone (L) | | | |
| Radio | | | |
| Drinking | | | |
| Reaching | | | |
| Makeup | | | |
| Passenger | | | |

### 4. Voice Assistant (Gemma 3n)

| Metric | Baseline | Fine-tuned | Change |
|--------|----------|------------|--------|
| Intent Accuracy | | | |
| Response Quality | | | |
| Inference (ms) | | | |

---

## System-Level Metrics

### Latency Breakdown

| Component | Time (ms) |
|-----------|-----------|
| Image Preprocessing | |
| Pedestrian Detection | |
| Drowsiness Detection | |
| Distraction Detection | |
| Router Decision | |
| Voice Processing | |
| **Total Pipeline** | |

### Resource Usage

| Resource | Value |
|----------|-------|
| Total Model Parameters | |
| LoRA Parameters | |
| Memory (GPU) | |
| Memory (System) | |

---

## Why On-Device Matters

| Metric | Cloud | On-Device | Benefit |
|--------|-------|-----------|---------|
| Latency | 100-500ms | <50ms | 10x faster |
| Privacy | Data transmitted | Data local | 100% private |
| Offline | Fails | Works | Always available |
| Cost | $/inference | Free after deploy | $0 marginal |

---

## Key Achievements

- [ ] Pedestrian detection with ___% accuracy
- [ ] Drowsiness detection with ___% accuracy
- [ ] 10-class distraction detection with ___% accuracy
- [ ] Voice commands with ___% intent accuracy
- [ ] End-to-end latency under 100ms
- [ ] Agentic routing with priority-based decisions
- [ ] Live demo working

---

## Technical Innovation

1. **Multi-Model Routing**: Priority-based system activates only needed models
2. **LoRA Efficiency**: Trained ~1% of parameters, achieved XX% improvement
3. **Truly Multimodal**: Vision (external) + Vision (internal) + Audio
4. **Agentic Behavior**: Not just detection, but contextual actions

---

## Files to Include in Presentation

- [ ] `models/*/metrics/baseline.json` - All baseline metrics
- [ ] `models/*/metrics/finetuned.json` - All fine-tuned metrics
- [ ] `presentation/screenshots/` - Demo screenshots
- [ ] `presentation/metrics/confusion_matrices/` - Per-model confusion matrices
- [ ] Training loss curves for each model
- [ ] Architecture diagram (high resolution)
