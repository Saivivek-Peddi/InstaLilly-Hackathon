#!/usr/bin/env python3
"""
Corrected Baseline Evaluation - PaliGemma2 448
Tests on 100 sample images with proper response parsing
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
BATCH_SIZE = 16
NUM_WORKERS = 4
MAX_SAMPLES = 100  # Evaluate on 100 images


class EvalDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        # Simple prompt for detection
        self.prompt = "detect pedestrian\n"
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        
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
    max_len = max(item["input_ids"].shape[0] for item in batch)
    
    input_ids = []
    attention_mask = []
    pixel_values = []
    labels = []
    paths = []
    
    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len
        
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


def extract_response(full_output: str, prompt: str = "detect pedestrian") -> str:
    """Extract only the generated response, not the prompt."""
    # Remove the prompt from output
    if prompt in full_output:
        response = full_output.split(prompt)[-1].strip()
    else:
        response = full_output.strip()
    return response


def map_response(response: str) -> str:
    """Map model response to class label."""
    r = response.lower().strip()
    
    # Check for multiple pedestrians keywords
    if "multiple" in r or "several" in r or "many" in r or r.startswith("2") or r.startswith("3"):
        return "multiple_pedestrians"
    # Check for no pedestrian keywords
    elif "no" == r or "none" in r or "zero" in r or r == "0" or "no pedestrian" in r or "no_pedestrian" in r:
        return "no_pedestrian"
    # Check for single pedestrian
    elif "yes" in r or "pedestrian" in r or "person" in r or r == "1" or "single" in r:
        return "pedestrian"
    
    # Default based on first word
    first_word = r.split()[0] if r else ""
    if first_word in ["no", "none", "0", "zero"]:
        return "no_pedestrian"
    elif first_word in ["yes", "1", "one", "single"]:
        return "pedestrian"
    
    return "unknown"


def run_baseline():
    print("="*60)
    print(f"🚀 PALIGEMMA2-3B-PT-448 BASELINE (100 samples)")
    print("="*60)
    
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(DATA_DIR / "prompts.csv")
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    
    # Sample 100 images with stratification
    if len(test_df) > MAX_SAMPLES:
        test_df = test_df.groupby('label').apply(
            lambda x: x.sample(n=min(len(x), MAX_SAMPLES//3), random_state=42)
        ).reset_index(drop=True)
    
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
    
    # Quick visual test
    print("\n🧪 Quick test on 3 samples:")
    prompt = "detect pedestrian\n"
    for i, row in test_df.head(3).iterrows():
        image = Image.open(row["path"]).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        full_resp = processor.decode(out[0], skip_special_tokens=True)
        extracted = extract_response(full_resp)
        predicted = map_response(extracted)
        print(f"  {row['label']:20} → pred={predicted:20} | raw='{extracted[:40]}'")
    
    # Full evaluation
    print(f"\n⚡ Running evaluation (batch={BATCH_SIZE})...")
    dataset = EvalDataset(test_df, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
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
                max_new_tokens=50,
                do_sample=False,
            )
            batch_time = (time.time() - batch_start) * 1000
            
            responses = processor.batch_decode(outputs, skip_special_tokens=True)
            
            for i, full_resp in enumerate(responses):
                extracted = extract_response(full_resp)
                pred_class = map_response(extracted)
                true_label = labels[i]
                is_correct = pred_class == true_label
                if is_correct:
                    correct += 1
                total += 1
                
                predictions.append({
                    "path": paths[i],
                    "true_label": true_label,
                    "predicted": pred_class,
                    "extracted_response": extracted[:100],
                    "full_response": full_resp[:200],
                    "correct": is_correct,
                    "inference_time_ms": batch_time / len(labels)
                })
    
    total_time = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    avg_latency = (total_time * 1000) / total if total > 0 else 0
    
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Latency: {avg_latency:.2f} ms/image")
    print(f"Throughput: {total/total_time:.1f} img/s")
    
    pred_df = pd.DataFrame(predictions)
    print("\nPer-class accuracy:")
    for label in ["no_pedestrian", "pedestrian", "multiple_pedestrians"]:
        cdf = pred_df[pred_df["true_label"] == label]
        if len(cdf) > 0:
            print(f"  {label}: {cdf['correct'].mean():.4f} ({int(cdf['correct'].sum())}/{len(cdf)})")
    
    print("\nSample predictions:")
    for _, row in pred_df.head(10).iterrows():
        s = "✓" if row["correct"] else "✗"
        print(f"  {s} {row['true_label']:20} → {row['predicted']:20} | '{row['extracted_response'][:30]}'")
    
    # Save metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model": BASE_MODEL,
        "resolution": "448x448",
        "num_samples": total,
        "accuracy": accuracy,
        "per_class": {
            l: float(pred_df[pred_df["true_label"]==l]["correct"].mean()) 
            for l in pred_df["true_label"].unique() if len(pred_df[pred_df["true_label"]==l]) > 0
        },
        "latency_ms": avg_latency,
        "throughput_img_per_s": total/total_time,
    }
    
    with open(METRICS_DIR / "baseline_448_100.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pred_df.to_csv(METRICS_DIR / "baseline_448_100_predictions.csv", index=False)
    
    print(f"\n✅ Saved to {METRICS_DIR}/baseline_448_100.json")
    return metrics


if __name__ == "__main__":
    run_baseline()
