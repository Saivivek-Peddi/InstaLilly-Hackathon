#!/usr/bin/env python3
"""
Training Script for Distraction Detection Model

Uses PaliGemma 2 with LoRA fine-tuning.

Run on VM:
    cd models/distraction_detector
    python train.py

Options:
    --epochs: Number of training epochs (default: 2)
    --batch_size: Batch size (default: 4)
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
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML dependencies not available. Run on VM with: uv sync --extra ml")

# Paths
MODEL_DIR = Path(__file__).parent
DATA_DIR = MODEL_DIR / "data" / "processed"
CHECKPOINTS_DIR = MODEL_DIR / "checkpoints"
METRICS_DIR = MODEL_DIR / "metrics"

# Model config - Using PaliGemma 2
BASE_MODEL = "google/paligemma2-3b-pt-448"

# Classes (5 classes based on actual data)
CLASSES = [
    "safe_driving", "texting_phone", "talking_phone",
    "other_activities", "turning"
]

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention only
    "task_type": "CAUSAL_LM"
}


class DistractionDataset(Dataset):
    """Dataset for distraction detection fine-tuning with PaliGemma2."""

    def __init__(self, df: pd.DataFrame, processor, max_length: int = 1200):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length
        self.prompt = "Driver activity?\n"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image = Image.open(row["path"]).convert("RGB")

        # Use suffix parameter - this is the key for proper label creation!
        inputs = self.processor(
            text=self.prompt,
            images=image,
            suffix=row["label"],  # e.g., "safe_driving", "texting_right", etc.
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length
        )

        result = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
        }
        if "token_type_ids" in inputs:
            result["token_type_ids"] = inputs["token_type_ids"].squeeze(0)
        if "labels" in inputs:
            result["labels"] = inputs["labels"].squeeze(0)
        return result


def get_balanced_sample(df, samples_per_class=200):
    """Get balanced sample across all classes."""
    balanced = []
    for label in CLASSES:
        class_df = df[df["label"] == label]
        n_samples = min(samples_per_class, len(class_df))
        if n_samples > 0:
            balanced.append(class_df.sample(n=n_samples, random_state=42))
    result = pd.concat(balanced, ignore_index=True)
    # Shuffle
    return result.sample(frac=1, random_state=42).reset_index(drop=True)


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

    print(f"Data loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    return train_df, val_df, test_df


def setup_model(use_4bit: bool = True):
    """Setup PaliGemma2 with LoRA."""
    print(f"Loading {BASE_MODEL}...")

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

    # Load processor - use PaliGemmaProcessor for proper suffix handling
    processor = PaliGemmaProcessor.from_pretrained(BASE_MODEL)

    # Load model
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Freeze vision encoder - freeze all parameters with "vision" in name
    print("Freezing vision encoder...")
    frozen_count = 0
    for name, param in model.named_parameters():
        if "vision" in name:
            param.requires_grad = False
            frozen_count += 1
    print(f"  Froze {frozen_count} vision parameters")

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
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    return model, processor


def evaluate_model(model, processor, test_df, device="cuda", max_samples=100):
    """Evaluate model on test set."""
    print(f"\nEvaluating model on {min(max_samples, len(test_df))} samples...")

    model.eval()
    correct = 0
    total = 0
    predictions = []
    prompt = "Driver activity?\n"

    # Sample if too many
    eval_df = test_df.sample(n=min(max_samples, len(test_df)), random_state=42)

    with torch.no_grad():
        for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
            try:
                image = Image.open(row["path"]).convert("RGB")

                inputs = processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )

                # Decode full response and extract the answer part
                response = processor.decode(outputs[0], skip_special_tokens=True)
                # Get part after the prompt
                answer = response.split("?")[-1].strip().lower()

                # Match to classes
                pred_class = None
                for cls in CLASSES:
                    if cls in answer or cls.replace("_", " ") in answer:
                        pred_class = cls
                        break

                # If no exact match, check first word
                if pred_class is None and answer:
                    first_word = answer.split()[0] if answer.split() else ""
                    for cls in CLASSES:
                        if cls.startswith(first_word) or first_word in cls:
                            pred_class = cls
                            break

                predictions.append({
                    "true_label": row["label"],
                    "predicted": pred_class,
                    "response": answer[:30],
                    "correct": pred_class == row["label"]
                })

                if pred_class == row["label"]:
                    correct += 1
                total += 1

            except Exception as e:
                print(f"  Error: {e}")
                continue

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")

    return accuracy, predictions


def train(args):
    """Main training function."""
    if not ML_AVAILABLE:
        print("ML dependencies not available. Run on VM.")
        return

    print("="*60)
    print("Distraction Detection Model Training")
    print("="*60)

    # Create directories
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, val_df, test_df = load_data()

    # Setup model
    model, processor = setup_model(use_4bit=not args.no_4bit)

    # Run baseline evaluation first
    print("\nRunning baseline evaluation (before fine-tuning)...")
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

    print(f"Baseline accuracy: {baseline_acc:.2%}")

    if args.eval_only:
        return

    # Create datasets
    print("\nCreating datasets...")
    # Use balanced sampling: 200 per class = 1000 total (5 classes)
    balanced_train = get_balanced_sample(train_df, samples_per_class=200)
    print(f"Balanced training set: {len(balanced_train)} samples")
    print(f"Class distribution: {balanced_train['label'].value_counts().to_dict()}")

    train_dataset = DistractionDataset(balanced_train, processor)
    val_dataset = DistractionDataset(val_df.head(200), processor)

    # Training arguments - checkpoint every 25 steps (~100 samples with batch 4)
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,  # No accumulation for faster checkpoints
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,  # Eval every 25 steps (~100 samples)
        save_strategy="steps",
        save_steps=25,  # Save every 25 steps
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Save model
    print("\nSaving model...")
    trainer.save_model(str(CHECKPOINTS_DIR / "final"))
    processor.save_pretrained(str(CHECKPOINTS_DIR / "final"))

    # Final evaluation
    print("\nFinal evaluation on test set...")
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

    # Calculate improvement
    improvement = final_acc - baseline_acc

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Baseline Accuracy:  {baseline_acc:.2%}")
    print(f"Finetuned Accuracy: {final_acc:.2%}")
    print(f"Improvement:        +{improvement:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Train Distraction Detection Model")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate base model")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
