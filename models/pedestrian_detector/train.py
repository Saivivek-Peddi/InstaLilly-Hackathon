#!/usr/bin/env python3
"""
Training Script for Pedestrian Detection Model

Uses PaliGemma 2 with LoRA fine-tuning for pedestrian detection from dashcam images.

Run on VM:
    cd models/pedestrian_detector
    python train.py

Options:
    --epochs: Number of training epochs (default: 3)
    --batch_size: Batch size (default: 8)
    --lr: Learning rate (default: 2e-4)
    --eval_only: Only run evaluation on base model
    --baseline: Run baseline evaluation before training
"""

import os
import json
import argparse
import time
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
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "task_type": "CAUSAL_LM"
}


class PedestrianDataset(Dataset):
    """Dataset for pedestrian detection fine-tuning."""

    def __init__(self, df: pd.DataFrame, processor, max_length: int = 256):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image = Image.open(row["path"]).convert("RGB")

        # Create prompt and response
        prompt = row["prompt"]
        response = row["response"]

        # Process inputs with suffix (response) for training
        # PaliGemma processor handles image tokens automatically
        inputs = self.processor(
            text=prompt,
            images=image,
            suffix=response,  # This creates proper labels
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        # Squeeze batch dimension
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "token_type_ids": inputs["token_type_ids"].squeeze(0),
            "labels": inputs["labels"].squeeze(0) if "labels" in inputs else inputs["input_ids"].squeeze(0)
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


def setup_model(use_4bit: bool = True, freeze_vision: bool = True):
    """Setup PaliGemma with LoRA and frozen vision tower."""
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

    # Freeze vision tower (keeps visual features fixed, only trains language model)
    if freeze_vision:
        print("🧊 Freezing vision tower...")
        vision_tower_frozen = 0

        # Freeze vision tower parameters
        for name, param in model.named_parameters():
            if "vision_tower" in name or "vision_model" in name or "image_encoder" in name:
                param.requires_grad = False
                vision_tower_frozen += param.numel()

        # Also freeze the multi-modal projector if you want (optional)
        # for name, param in model.named_parameters():
        #     if "multi_modal_projector" in name:
        #         param.requires_grad = False

        print(f"   Frozen {vision_tower_frozen:,} vision parameters")

    # Setup LoRA (only applies to unfrozen layers)
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
    frozen_params = total_params - trainable_params

    print(f"📊 Model parameters:")
    print(f"   Total:     {total_params:,}")
    print(f"   Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"   Frozen:    {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")

    return model, processor


def evaluate_model(model, processor, test_df, device="cuda", batch_size=16, num_workers=4, max_samples=None):
    """Evaluate model on test set with batched inference."""
    print("\n📊 Evaluating model...")

    if max_samples:
        test_df = test_df.head(max_samples)
        print(f"   Using {max_samples} samples for evaluation")

    model.eval()
    predictions = []
    classes = ["no_pedestrian", "pedestrian", "multiple_pedestrians"]

    # Create evaluation dataset
    class EvalDataset(Dataset):
        def __init__(self, df, processor):
            self.df = df.reset_index(drop=True)
            self.processor = processor

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image = Image.open(row["path"]).convert("RGB")
            inputs = self.processor(
                text=row["prompt"],
                images=image,
                return_tensors="pt",
                padding="max_length",
                max_length=256,
                truncation=True
            )
            return {
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "pixel_values": inputs["pixel_values"].squeeze(),
                "label": row["label"],
                "idx": idx
            }

    eval_dataset = EvalDataset(test_df, processor)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    correct = 0
    total = 0
    total_time = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"]
            indices = batch["idx"]

            # Measure inference time
            start_time = time.time()
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=100,
                do_sample=False
            )
            batch_time = (time.time() - start_time) * 1000  # ms
            total_time += batch_time

            # Decode responses
            responses = processor.batch_decode(outputs, skip_special_tokens=True)

            for i, response in enumerate(responses):
                response_lower = response.lower()
                true_label = labels[i]

                # Extract predicted class
                pred_class = None
                if "multiple_pedestrian" in response_lower or "multiple pedestrian" in response_lower:
                    pred_class = "multiple_pedestrians"
                elif "no_pedestrian" in response_lower or "no pedestrian" in response_lower:
                    pred_class = "no_pedestrian"
                elif "pedestrian" in response_lower:
                    pred_class = "pedestrian"

                is_correct = pred_class == true_label
                if is_correct:
                    correct += 1
                total += 1

                predictions.append({
                    "true_label": true_label,
                    "predicted": pred_class,
                    "response": response[:200],  # Truncate for storage
                    "correct": is_correct,
                    "inference_time_ms": batch_time / len(responses)
                })

    accuracy = correct / total if total > 0 else 0
    avg_inference_time = total_time / total if total > 0 else 0

    print(f"✅ Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"⏱️  Avg inference time: {avg_inference_time:.2f} ms/image")
    print(f"⏱️  Total evaluation time: {total_time/1000:.1f}s")

    return accuracy, avg_inference_time, predictions


def train(args):
    """Main training function."""
    if not ML_AVAILABLE:
        print("❌ ML dependencies not available. Run on VM.")
        return

    print("="*60)
    print("🚶 Pedestrian Detection Model Training")
    print("="*60)

    # Create directories
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, val_df, test_df = load_data()

    # Setup model
    model, processor = setup_model(
        use_4bit=not args.no_4bit,
        freeze_vision=not args.unfreeze_vision
    )

    # Baseline evaluation
    if args.eval_only or args.baseline:
        print("\n📊 Running baseline evaluation...")
        baseline_acc, baseline_latency, baseline_preds = evaluate_model(
            model, processor, test_df,
            batch_size=args.eval_batch_size,
            max_samples=args.max_eval_samples
        )

        # Save baseline metrics
        baseline_metrics = {
            "timestamp": datetime.now().isoformat(),
            "model": BASE_MODEL,
            "accuracy": baseline_acc,
            "inference_time_ms": baseline_latency,
            "num_samples": len(test_df)
        }
        with open(METRICS_DIR / "baseline.json", "w") as f:
            json.dump(baseline_metrics, f, indent=2)

        # Save predictions
        pd.DataFrame(baseline_preds).to_csv(METRICS_DIR / "baseline_predictions.csv", index=False)

        print(f"✅ Baseline metrics saved to {METRICS_DIR / 'baseline.json'}")

        if args.eval_only:
            return

    # Create datasets
    train_dataset = PedestrianDataset(train_df, processor)
    val_dataset = PedestrianDataset(val_df, processor)

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
    train_start = time.time()
    train_result = trainer.train()
    train_time = (time.time() - train_start) / 3600  # hours

    # Save model
    print("\n💾 Saving model...")
    trainer.save_model(str(CHECKPOINTS_DIR / "final"))
    processor.save_pretrained(str(CHECKPOINTS_DIR / "final"))

    # Final evaluation
    print("\n📊 Final evaluation on test set...")
    final_acc, final_latency, final_preds = evaluate_model(
        model, processor, test_df,
        batch_size=args.eval_batch_size,
        max_samples=args.max_eval_samples
    )

    # Load baseline for comparison
    baseline_file = METRICS_DIR / "baseline.json"
    baseline_acc = 0
    if baseline_file.exists():
        with open(baseline_file, "r") as f:
            baseline_data = json.load(f)
            baseline_acc = baseline_data.get("accuracy", 0)

    improvement = final_acc - baseline_acc

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
        "baseline_accuracy": baseline_acc,
        "final_accuracy": final_acc,
        "improvement": improvement,
        "improvement_percent": (improvement / baseline_acc * 100) if baseline_acc > 0 else 0,
        "inference_time_ms": final_latency,
        "train_loss": train_result.training_loss,
        "training_time_hours": train_time,
        "num_samples": len(test_df)
    }
    with open(METRICS_DIR / "finetuned.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Save predictions
    pd.DataFrame(final_preds).to_csv(METRICS_DIR / "predictions.csv", index=False)

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"Final Accuracy:    {final_acc:.4f}")
    print(f"Improvement:       +{improvement:.4f} ({improvement/baseline_acc*100:.1f}%)" if baseline_acc > 0 else f"Final: {final_acc:.4f}")
    print(f"Inference Time:    {final_latency:.2f} ms")
    print(f"Training Time:     {train_time:.2f} hours")
    print(f"Checkpoints:       {CHECKPOINTS_DIR}")
    print(f"Metrics:           {METRICS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train Pedestrian Detection Model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate base model")
    parser.add_argument("--baseline", action="store_true", help="Run baseline eval before training")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--unfreeze_vision", action="store_true", help="Don't freeze vision tower (not recommended)")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Max samples for evaluation (for quick testing)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
