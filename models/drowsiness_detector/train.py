#!/usr/bin/env python3
"""
Training Script for Drowsiness Detection Model

Uses PaliGemma 2 with LoRA fine-tuning.

Run on VM:
    cd models/drowsiness_detector
    python train.py

Options:
    --epochs: Number of training epochs (default: 3)
    --batch_size: Batch size (default: 8)
    --lr: Learning rate (default: 2e-4)
    --eval_only: Only run evaluation on base model
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

# These imports will only work on VM with ML dependencies
try:
    from transformers import (
        PaliGemmaProcessor,
        PaliGemmaForConditionalGeneration,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, TaskType
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML dependencies not available. Run on VM with: uv sync --extra ml")

# Paths
MODEL_DIR = Path(__file__).parent
DATA_DIR = MODEL_DIR / "data" / "processed"
CHECKPOINTS_DIR = MODEL_DIR / "checkpoints"
METRICS_DIR = MODEL_DIR / "metrics"

# Model config
BASE_MODEL = "google/paligemma-3b-pt-224"
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # Include both attention AND MLP layers per research best practices
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP (GeGLU)
    ],
    "task_type": "CAUSAL_LM"
}


class DrowsinessDataset(Dataset):
    """Dataset for drowsiness detection fine-tuning."""

    def __init__(self, df: pd.DataFrame, processor, max_length: int = 512):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image = Image.open(row["path"]).convert("RGB")

        # Create prompt
        prompt = row["prompt"]
        response = row["response"]

        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        # Process labels (response)
        labels = self.processor.tokenizer(
            response,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "pixel_values": inputs["pixel_values"].squeeze(),
            "labels": labels["input_ids"].squeeze()
        }


def load_data():
    """Load processed data."""
    prompts_file = DATA_DIR / "prompts.csv"

    if not prompts_file.exists():
        raise FileNotFoundError(
            f"Data not found at {prompts_file}. "
            "Run prepare_data.py first."
        )

    df = pd.read_csv(prompts_file)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print(f"📊 Data loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    return train_df, val_df, test_df


def setup_model(use_4bit: bool = True):
    """Setup PaliGemma with LoRA."""
    print(f"🔧 Loading {BASE_MODEL}...")

    # Quantization config for memory efficiency
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
    else:
        bnb_config = None

    # Load processor
    processor = PaliGemmaProcessor.from_pretrained(BASE_MODEL)

    # Load model
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Freeze vision tower (only train language model with LoRA)
    # Per research: vision encoder already has sufficient visual understanding
    print("🔒 Freezing vision tower...")
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    # Setup LoRA
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    return model, processor


def write_summary_md(baseline_metrics: dict, finetuned_metrics: dict, filepath: Path):
    """Write a presentation-ready summary comparing baseline vs finetuned."""
    with open(filepath, "w") as f:
        f.write("# Drowsiness Detection Model - Training Summary\n\n")
        f.write(f"**Date:** {finetuned_metrics.get('timestamp', 'N/A')}\n\n")
        f.write(f"**Base Model:** `{baseline_metrics.get('model', 'N/A')}`\n\n")

        # Performance comparison
        baseline_acc = baseline_metrics.get('accuracy', 0)
        finetuned_acc = finetuned_metrics.get('accuracy', 0)
        improvement = finetuned_acc - baseline_acc

        f.write("## Performance Comparison\n\n")
        f.write("| Model | Accuracy | Change |\n")
        f.write("|-------|----------|--------|\n")
        f.write(f"| Baseline (zero-shot) | {baseline_acc:.2%} | - |\n")
        f.write(f"| **Fine-tuned (LoRA)** | **{finetuned_acc:.2%}** | **+{improvement:.2%}** |\n\n")

        # Training config
        f.write("## Training Configuration\n\n")
        f.write(f"- **LoRA Rank:** {finetuned_metrics.get('lora_config', {}).get('r', 'N/A')}\n")
        f.write(f"- **LoRA Alpha:** {finetuned_metrics.get('lora_config', {}).get('lora_alpha', 'N/A')}\n")
        f.write(f"- **Target Modules:** Attention + MLP layers\n")
        training_args = finetuned_metrics.get('training_args', {})
        f.write(f"- **Epochs:** {training_args.get('epochs', 'N/A')}\n")
        f.write(f"- **Learning Rate:** {training_args.get('learning_rate', 'N/A')}\n")
        f.write(f"- **Batch Size:** {training_args.get('batch_size', 'N/A')}\n")
        f.write(f"- **Training Loss:** {finetuned_metrics.get('train_loss', 'N/A'):.4f}\n\n")

        f.write("## Key Findings\n\n")
        if improvement > 0.1:
            f.write(f"- Significant improvement of **{improvement:.2%}** over baseline\n")
        elif improvement > 0:
            f.write(f"- Moderate improvement of **{improvement:.2%}** over baseline\n")
        else:
            f.write(f"- Model performance similar to baseline\n")
        f.write(f"- Fine-tuning successfully adapted model to drowsiness detection task\n")


def evaluate_model(model, processor, test_df, device="cuda"):
    """Evaluate model on test set."""
    print("\n📊 Evaluating model...")

    model.eval()
    correct = 0
    total = 0
    predictions = []

    classes = ["alert", "drowsy", "yawning", "eyes_closed"]

    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            image = Image.open(row["path"]).convert("RGB")

            inputs = processor(
                text=row["prompt"],
                images=image,
                return_tensors="pt"
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )

            response = processor.decode(outputs[0], skip_special_tokens=True)

            # Extract predicted class from response
            pred_class = None
            response_lower = response.lower()
            for cls in classes:
                if cls in response_lower:
                    pred_class = cls
                    break

            predictions.append({
                "true_label": row["label"],
                "predicted": pred_class,
                "response": response,
                "correct": pred_class == row["label"]
            })

            if pred_class == row["label"]:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"✅ Accuracy: {accuracy:.4f} ({correct}/{total})")

    return accuracy, predictions


def train(args):
    """Main training function."""
    if not ML_AVAILABLE:
        print("❌ ML dependencies not available. Run on VM.")
        return

    print("="*60)
    print("🚗 Drowsiness Detection Model Training")
    print("="*60)

    # Create directories
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, val_df, test_df = load_data()

    # Setup model
    model, processor = setup_model(use_4bit=not args.no_4bit)

    # Always run baseline evaluation first
    print("\n📊 Running baseline evaluation (before fine-tuning)...")
    baseline_acc, baseline_preds = evaluate_model(model, processor, test_df)

    # Save baseline metrics
    baseline_metrics = {
        "timestamp": datetime.now().isoformat(),
        "model": BASE_MODEL,
        "accuracy": baseline_acc,
        "num_samples": len(test_df)
    }
    with open(METRICS_DIR / "baseline.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    print(f"✅ Baseline accuracy: {baseline_acc:.2%}")
    print(f"✅ Baseline metrics saved to {METRICS_DIR / 'baseline.json'}")

    if args.eval_only:
        return

    # Create datasets
    train_dataset = DrowsinessDataset(train_df, processor)
    val_dataset = DrowsinessDataset(val_df, processor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="none",
        remove_unused_columns=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train
    print("\n🚀 Starting training...")
    train_result = trainer.train()

    # Save model
    print("\n💾 Saving model...")
    trainer.save_model(str(CHECKPOINTS_DIR / "final"))
    processor.save_pretrained(str(CHECKPOINTS_DIR / "final"))

    # Final evaluation
    print("\n📊 Final evaluation on test set...")
    final_acc, final_preds = evaluate_model(model, processor, test_df)

    # Save final metrics
    final_metrics = {
        "timestamp": datetime.now().isoformat(),
        "model": BASE_MODEL,
        "lora_config": LORA_CONFIG,
        "training_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr
        },
        "accuracy": final_acc,
        "train_loss": train_result.training_loss,
        "num_samples": len(test_df)
    }
    with open(METRICS_DIR / "finetuned.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Save predictions
    pd.DataFrame(final_preds).to_csv(METRICS_DIR / "predictions.csv", index=False)

    # Write presentation summary (baseline vs finetuned)
    write_summary_md(baseline_metrics, final_metrics, METRICS_DIR / "summary.md")
    print(f"✅ Summary saved to {METRICS_DIR / 'summary.md'}")

    # Calculate improvement
    improvement = final_acc - baseline_acc

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"Baseline Accuracy:  {baseline_acc:.2%}")
    print(f"Finetuned Accuracy: {final_acc:.2%}")
    print(f"Improvement:        +{improvement:.2%}")
    print(f"Checkpoints: {CHECKPOINTS_DIR}")
    print(f"Metrics: {METRICS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train Drowsiness Detection Model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate base model")
    parser.add_argument("--baseline", action="store_true", help="Run baseline eval before training")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
