# Model Metrics Tracking

## Instructions
Fill this template for each model at each stage (baseline → fine-tuned → deployed)

---

## Model: [MODEL_NAME]

### Baseline Metrics (Before Fine-tuning)

| Metric | Value |
|--------|-------|
| **Model Name** | |
| **Base Model** | PaliGemma-3B / Gemma-3n |
| **Task** | |
| **Test Dataset Size** | |
| **Date Evaluated** | |

#### Accuracy Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Class 1 | | | | |
| Class 2 | | | | |
| ... | | | | |
| **Overall** | | | | |

#### Performance Metrics

| Metric | Value |
|--------|-------|
| Inference Time (avg ms) | |
| Inference Time (p95 ms) | |
| Inference Time (p99 ms) | |
| Model Size (MB) | |
| GPU Memory Usage (MB) | |

---

### Fine-tuned Metrics (After LoRA)

| Metric | Value |
|--------|-------|
| **Training Date** | |
| **Training Duration** | |
| **LoRA Rank (r)** | 16 |
| **LoRA Alpha** | 32 |
| **Learning Rate** | |
| **Epochs** | |
| **Trainable Parameters** | |
| **% of Total Parameters** | |

#### Accuracy Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Class 1 | | | | |
| Class 2 | | | | |
| ... | | | | |
| **Overall** | | | | |

#### Performance Metrics

| Metric | Value |
|--------|-------|
| Inference Time (avg ms) | |
| Model Size with LoRA (MB) | |
| LoRA Adapter Size (MB) | |

---

### Improvement Summary

| Metric | Baseline | Fine-tuned | Delta | % Change |
|--------|----------|------------|-------|----------|
| Accuracy | | | | |
| Precision | | | | |
| Recall | | | | |
| F1-Score | | | | |
| Inference Time | | | | |

---

### Sample Predictions

#### Correct Predictions
| Image | Ground Truth | Prediction | Confidence |
|-------|--------------|------------|------------|
| [img1] | | | |
| [img2] | | | |

#### Incorrect Predictions (for analysis)
| Image | Ground Truth | Prediction | Confidence | Analysis |
|-------|--------------|------------|------------|----------|
| [img1] | | | | |

---

### Confusion Matrix

```
Paste confusion matrix visualization or data here
```

---

### Training Curves

- [ ] Loss curve saved to: `checkpoints/training_loss.png`
- [ ] Accuracy curve saved to: `checkpoints/accuracy_curve.png`
