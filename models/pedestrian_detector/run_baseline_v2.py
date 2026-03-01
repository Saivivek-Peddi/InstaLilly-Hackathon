#!/usr/bin/env python3
"""
Proper Baseline - PaliGemma2 448 (1024 image tokens)
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

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

MODEL_DIR = Path(__file__).parent
DATA_DIR = MODEL_DIR / "data" / "processed"
METRICS_DIR = MODEL_DIR / "metrics"

BASE_MODEL = "google/paligemma2-3b-pt-448"
BATCH_SIZE = 32
NUM_WORKERS = 8


class EvalDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        # Proper prompt with image token and newline
        self.prompt = "<image>answer en Is there a pedestrian? yes, no, or multiple\n"
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        
        # Don't use max_length truncation - let processor handle it
        inputs = self.processor(
            text=self.prompt,
            images=image,
            return_tensors="pt",
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "pixel_values": inputs["pixel_values"].squeeze(),
            "label": row["label"],
            "path": row["path"]
        }


def collate_fn(batch):
    """Custom collate to handle variable length sequences."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    
    input_ids = []
    attention_mask = []
    pixel_values = []
    labels = []
    paths = []
    
    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len
        
        # Pad input_ids and attention_mask
        if pad_len > 0:
            input_ids.append(torch.cat([
                item["input_ids"],
                torch.zeros(pad_len, dtype=item["input_ids"].dtype)
            ]))
            attention_mask.append(torch.cat([
                item["attention_mask"],
                torch.zeros(pad_len, dtype=item["attention_mask"].dtype)
            ]))
        else:
            input_ids.append(item["input_ids"])
            attention_mask.append(item["attention_mask"])
        
        pixel_values.append(item["pixel_values"])
        labels.append(item["label"])
        paths.append(item["path"])
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "pixel_values": torch.stack(pixel_values),
        "label": labels,
        "path": paths
    }


def map_response(response: str) -> str:
    r = response.lower().strip()
    if "multiple" in r or "several" in r or "many" in r:
        return "multiple_pedestrians"
    elif "no" in r or "none" in r or "zero" in r or "not" in r:
        return "no_pedestrian"
    elif "yes" in r or "pedestrian" in r or "person" in r or "people" in r:
        return "pedestrian"
    return None


def run_baseline():
    print("="*60)
    print("🚀 PALIGEMMA2-3B-PT-448 BASELINE")
    print("="*60)
    
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(DATA_DIR / "prompts.csv")
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"📊 Test samples: {len(test_df)}")
    print(f"   Distribution: {test_df['label'].value_counts().to_dict()}")
    
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
    )
    model.eval()
    print("✅ Model loaded")
    
    # Quick test on 3 samples
    print("\n🧪 Quick test on samples:")
    for i, row in test_df.head(3).iterrows():
        image = Image.open(row["path"]).convert("RGB")
        prompt = "<image>answer en Is there a pedestrian? yes, no, or multiple\n"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        resp = processor.decode(out[0], skip_special_tokens=True)
        print(f"  {row['label']:20} → {resp[-50:]}")
    
    # Full evaluation
    print(f"\n⚡ Running full evaluation (batch={BATCH_SIZE})...")
    dataset = EvalDataset(test_df, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2,
    )
    
    predictions = []
    correct = 0
    total = 0
    start_time = time.time()
    
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
                max_new_tokens=30,
                do_sample=False,
            )
            batch_time = (time.time() - batch_start) * 1000
            
            responses = processor.batch_decode(outputs, skip_special_tokens=True)
            
            for i, response in enumerate(responses):
                pred_class = map_response(response)
                true_label = labels[i]
                is_correct = pred_class == true_label
                if is_correct:
                    correct += 1
                total += 1
                
                predictions.append({
                    "path": paths[i],
                    "true_label": true_label,
                    "predicted": pred_class,
                    "response": response[-80:],
                    "correct": is_correct,
                    "inference_time_ms": batch_time / len(labels)
                })
    
    total_time = time.time() - start_time
    accuracy = correct / total
    avg_latency = (total_time * 1000) / total
    
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Latency: {avg_latency:.2f} ms/image")
    print(f"Throughput: {total/total_time:.1f} img/s")
    
    pred_df = pd.DataFrame(predictions)
    print("\nPer-class:")
    for label in ["no_pedestrian", "pedestrian", "multiple_pedestrians"]:
        cdf = pred_df[pred_df["true_label"] == label]
        if len(cdf) > 0:
            print(f"  {label}: {cdf['correct'].mean():.4f}")
    
    print("\nSamples:")
    for _, row in pred_df.head(8).iterrows():
        s = "✓" if row["correct"] else "✗"
        print(f"  {s} {row['true_label']:20} → {row['predicted']} | {row['response'][:40]}")
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model": BASE_MODEL,
        "accuracy": accuracy,
        "per_class": {l: float(pred_df[pred_df["true_label"]==l]["correct"].mean()) 
                      for l in pred_df["true_label"].unique()},
        "latency_ms": avg_latency,
        "throughput": total/total_time,
        "samples": total,
    }
    
    with open(METRICS_DIR / "baseline.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pred_df.to_csv(METRICS_DIR / "baseline_predictions.csv", index=False)
    
    print(f"\n✅ Saved to {METRICS_DIR}")


if __name__ == "__main__":
    run_baseline()
