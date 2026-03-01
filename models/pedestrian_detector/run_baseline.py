#!/usr/bin/env python3
"""
Fast Baseline Evaluation - Parallelized across GPU

Uses:
- Large batch sizes (96GB VRAM available)
- Multiple DataLoader workers
- Mixed precision inference
- PaliGemma2 mix (instruction-tuned) for meaningful baseline
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig
)

# Config
MODEL_DIR = Path(__file__).parent
DATA_DIR = MODEL_DIR / "data" / "processed"
METRICS_DIR = MODEL_DIR / "metrics"

# Use instruction-tuned model for meaningful baseline
BASE_MODEL = "google/paligemma2-3b-mix-224"

# Optimized for 96GB VRAM
BATCH_SIZE = 32  # Large batch for parallel inference
NUM_WORKERS = 8  # Parallel data loading
PREFETCH_FACTOR = 4


class FastEvalDataset(Dataset):
    """Optimized dataset with pre-loaded images."""
    
    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.prompt = "Is there a pedestrian in this image? Answer: yes, no, or multiple."
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        
        inputs = self.processor(
            text=self.prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=128,  # Shorter for speed
            truncation=True
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "pixel_values": inputs["pixel_values"].squeeze(),
            "label": row["label"],
            "path": row["path"]
        }


def map_response_to_class(response: str) -> str:
    """Map model response to class label."""
    response = response.lower().strip()
    
    if "multiple" in response:
        return "multiple_pedestrians"
    elif "no" in response or "none" in response or "0" in response:
        return "no_pedestrian"
    elif "yes" in response or "pedestrian" in response or "person" in response:
        return "pedestrian"
    else:
        return None


def run_baseline():
    print("="*60)
    print("🚀 FAST BASELINE EVALUATION (Parallelized)")
    print("="*60)
    
    # Create metrics dir
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    prompts_file = DATA_DIR / "prompts.csv"
    df = pd.read_csv(prompts_file)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"📊 Test samples: {len(test_df)}")
    
    # Print class distribution
    print(f"   Class distribution: {test_df['label'].value_counts().to_dict()}")
    
    # Load model with optimization
    print(f"\n🔧 Loading {BASE_MODEL}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    processor = PaliGemmaProcessor.from_pretrained(BASE_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"  # Use scaled dot product attention
    )
    model.eval()
    
    print(f"✅ Model loaded on {next(model.parameters()).device}")
    
    # Create optimized dataloader
    dataset = FastEvalDataset(test_df, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True
    )
    
    print(f"\n⚡ Running inference with batch_size={BATCH_SIZE}, workers={NUM_WORKERS}")
    
    # Inference
    predictions = []
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            pixel_values = batch["pixel_values"].cuda()
            labels = batch["label"]
            paths = batch["path"]
            
            batch_start = time.time()
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=20,  # Short response needed
                do_sample=False,
                num_beams=1,  # Greedy for speed
            )
            
            batch_time = (time.time() - batch_start) * 1000
            per_sample_time = batch_time / len(labels)
            
            responses = processor.batch_decode(outputs, skip_special_tokens=True)
            
            for i, response in enumerate(responses):
                pred_class = map_response_to_class(response)
                true_label = labels[i]
                is_correct = pred_class == true_label
                
                if is_correct:
                    correct += 1
                total += 1
                
                predictions.append({
                    "path": paths[i],
                    "true_label": true_label,
                    "predicted": pred_class,
                    "response": response[:100],
                    "correct": is_correct,
                    "inference_time_ms": per_sample_time
                })
    
    total_time = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    avg_latency = (total_time * 1000) / total
    throughput = total / total_time
    
    # Results
    print("\n" + "="*60)
    print("📊 BASELINE RESULTS")
    print("="*60)
    print(f"Model: {BASE_MODEL}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Avg Latency: {avg_latency:.2f} ms/image")
    print(f"Throughput: {throughput:.1f} images/sec")
    print(f"Total Time: {total_time:.1f}s")
    
    # Per-class accuracy
    pred_df = pd.DataFrame(predictions)
    print("\nPer-class accuracy:")
    for label in ["no_pedestrian", "pedestrian", "multiple_pedestrians"]:
        class_df = pred_df[pred_df["true_label"] == label]
        if len(class_df) > 0:
            class_acc = class_df["correct"].mean()
            print(f"  {label}: {class_acc:.4f} ({class_df['correct'].sum()}/{len(class_df)})")
    
    # Save metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model": BASE_MODEL,
        "accuracy": accuracy,
        "per_class_accuracy": {
            label: float(pred_df[pred_df["true_label"] == label]["correct"].mean())
            for label in pred_df["true_label"].unique()
        },
        "inference_time_ms": avg_latency,
        "throughput_images_per_sec": throughput,
        "total_time_sec": total_time,
        "num_samples": total,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS
    }
    
    with open(METRICS_DIR / "baseline.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    pred_df.to_csv(METRICS_DIR / "baseline_predictions.csv", index=False)
    
    print(f"\n✅ Saved to {METRICS_DIR}/baseline.json")
    
    # Show sample predictions
    print("\n📋 Sample predictions:")
    for i, row in pred_df.head(5).iterrows():
        status = "✓" if row["correct"] else "✗"
        print(f"  {status} True: {row['true_label']:20} Pred: {str(row['predicted']):20} Response: {row['response'][:40]}")


if __name__ == "__main__":
    run_baseline()
