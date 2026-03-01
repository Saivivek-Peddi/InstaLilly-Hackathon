#!/usr/bin/env python3
import os, json, time
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
from transformers import (PaliGemmaProcessor, PaliGemmaForConditionalGeneration,
    TrainingArguments, Trainer, BitsAndBytesConfig, TrainerCallback)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_DIR = Path(__file__).parent
DATA_DIR = MODEL_DIR / "data" / "processed"
CHECKPOINTS_DIR = MODEL_DIR / "checkpoints"
METRICS_DIR = MODEL_DIR / "metrics"
BASE_MODEL = "google/paligemma2-3b-pt-448"
LORA_CONFIG = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]}

def get_gpu_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()/1024**3, torch.cuda.memory_reserved()/1024**3, torch.cuda.get_device_properties(0).total_memory/1024**3
    return 0,0,0

def print_gpu(prefix=""):
    a,r,t = get_gpu_mem()
    print(f"{prefix}GPU: {a:.1f}GB/{t:.0f}GB")

class ProgressCB(TrainerCallback):
    def __init__(self, total): self.total=total; self.start=None; self.pbar=None
    def on_train_begin(self, args, state, control, **kw):
        self.start=time.time(); self.pbar=tqdm(total=self.total, desc="Training", unit="step"); print_gpu("Start: ")
    def on_step_end(self, args, state, control, **kw):
        if self.pbar:
            self.pbar.update(1)
            if state.log_history:
                loss=state.log_history[-1].get("loss",0); a,_,t=get_gpu_mem()
                el=time.time()-self.start; sps=state.global_step/el if el>0 else 0
                eta=(self.total-state.global_step)/sps/60 if sps>0 else 0
                self.pbar.set_postfix({"loss":f"{loss:.4f}","GPU":f"{a:.0f}/{t:.0f}GB","ETA":f"{eta:.1f}m"})
    def on_train_end(self, args, state, control, **kw):
        if self.pbar: self.pbar.close()
        print(f"\nDone in {(time.time()-self.start)/60:.1f}min"); print_gpu("End: ")

class TrainDS(Dataset):
    def __init__(self, df, proc, maxlen=1200):
        self.df=df.reset_index(drop=True); self.proc=proc; self.maxlen=maxlen
        self.df["bin"]=self.df["label"].apply(lambda x: "yes" if x in ["pedestrian","multiple_pedestrians"] else "no")
        self.prompt="Is there a pedestrian?\n"
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row=self.df.iloc[i]; img=Image.open(row["path"]).convert("RGB")
        inp=self.proc(text=self.prompt, images=img, suffix=row["bin"], return_tensors="pt", padding="max_length", max_length=self.maxlen)
        r={"input_ids":inp["input_ids"].squeeze(0),"attention_mask":inp["attention_mask"].squeeze(0),"pixel_values":inp["pixel_values"].squeeze(0)}
        if "token_type_ids" in inp: r["token_type_ids"]=inp["token_type_ids"].squeeze(0)
        if "labels" in inp: r["labels"]=inp["labels"].squeeze(0)
        return r

class EvalDS(Dataset):
    def __init__(self, df, proc):
        self.df=df.reset_index(drop=True); self.proc=proc; self.prompt="Is there a pedestrian?\n"
        self.df["bin"]=self.df["label"].apply(lambda x: "yes" if x in ["pedestrian","multiple_pedestrians"] else "no")
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row=self.df.iloc[i]; img=Image.open(row["path"]).convert("RGB")
        inp=self.proc(text=self.prompt, images=img, return_tensors="pt")
        return {"input_ids":inp["input_ids"].squeeze(),"attention_mask":inp["attention_mask"].squeeze(),
                "pixel_values":inp["pixel_values"].squeeze(),"label":row["bin"],"path":row["path"]}

def collate(batch):
    ml=max(x["input_ids"].shape[0] for x in batch)
    ids,mask,pix,lab,pth=[],[],[],[],[]
    for x in batch:
        sl=x["input_ids"].shape[0]; pl=ml-sl
        ids.append(torch.cat([x["input_ids"],torch.zeros(pl,dtype=x["input_ids"].dtype)]) if pl>0 else x["input_ids"])
        mask.append(torch.cat([x["attention_mask"],torch.zeros(pl,dtype=x["attention_mask"].dtype)]) if pl>0 else x["attention_mask"])
        pix.append(x["pixel_values"]); lab.append(x["label"]); pth.append(x["path"])
    return {"input_ids":torch.stack(ids),"attention_mask":torch.stack(mask),"pixel_values":torch.stack(pix),"label":lab,"path":pth}

def evaluate(model, proc, df, bs=16, desc="Eval"):
    model.eval(); ds=EvalDS(df,proc); dl=DataLoader(ds,batch_size=bs,shuffle=False,num_workers=4,pin_memory=True,collate_fn=collate)
    preds=[]; correct=0; total=0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for b in tqdm(dl, desc=desc):
            out=model.generate(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda(),
                               pixel_values=b["pixel_values"].cuda(), max_new_tokens=10, do_sample=False)
            resps=proc.batch_decode(out, skip_special_tokens=True)
            for i,resp in enumerate(resps):
                r=resp.split("?")[-1].strip().lower()
                pred="yes" if "yes" in r or r.startswith("y") else ("no" if "no" in r or r.startswith("n") else "unk")
                ok=pred==b["label"][i]; correct+=int(ok); total+=1
                preds.append({"path":b["path"][i],"true":b["label"][i],"pred":pred,"resp":r[:30],"ok":ok})
    return correct/total if total else 0, preds

def main():
    print("="*70+"\n   PALIGEMMA2-448 PEDESTRIAN CLASSIFICATION FINETUNING\n"+"="*70)
    print_gpu("Init: ")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True); METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    df=pd.read_csv(DATA_DIR/"prompts.csv")
    train_df=df[df["split"]=="train"].reset_index(drop=True)
    val_df=df[df["split"]=="val"].reset_index(drop=True)
    test_df=df[df["split"]=="test"].reset_index(drop=True)
    print(f"\n[DATA] Train:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)}")
    
    print(f"\n[MODEL] Loading {BASE_MODEL}...")
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    proc=PaliGemmaProcessor.from_pretrained(BASE_MODEL)
    model=PaliGemmaForConditionalGeneration.from_pretrained(BASE_MODEL,quantization_config=bnb,device_map="auto",torch_dtype=torch.bfloat16)
    print_gpu("Loaded: ")
    
    print("[MODEL] Freezing vision...")
    for n,p in model.named_parameters():
        if "vision" in n: p.requires_grad=False
    
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=["q_proj","v_proj","k_proj","o_proj"],task_type=TaskType.CAUSAL_LM)
    model=get_peft_model(model,lora)
    tr=sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot=sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Trainable: {tr:,}/{tot:,} ({100*tr/tot:.2f}%)"); print_gpu("LoRA: ")
    
    print("\n"+"="*70+"\n   BASELINE (before training)\n"+"="*70)
    base_acc,base_preds=evaluate(model,proc,test_df.head(100),desc="Baseline")
    print(f"\n>>> BASELINE: {base_acc:.2%}")
    print("\nSamples (10):")
    for p in base_preds[:10]:
        s="OK" if p["ok"] else "X "
        print(f"  [{s}] {p['true']:3} -> {p['pred']:3} | {p['path'].split('/')[-1][:25]}")
    
    print("\n[TRAIN] Creating datasets...")
    train_ds=TrainDS(train_df,proc); val_ds=TrainDS(val_df,proc)
    bs,ga,ep=4,8,2; spe=len(train_df)//(bs*ga); tot_steps=spe*ep
    print(f"[TRAIN] {spe} steps/epoch x {ep} epochs = {tot_steps} total")
    
    args=TrainingArguments(output_dir=str(CHECKPOINTS_DIR),num_train_epochs=ep,per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,gradient_accumulation_steps=ga,learning_rate=2e-4,weight_decay=0.01,
        warmup_ratio=0.1,logging_steps=25,eval_strategy="steps",eval_steps=100,save_strategy="steps",
        save_steps=100,save_total_limit=2,load_best_model_at_end=True,bf16=True,report_to="none",
        remove_unused_columns=False,dataloader_num_workers=4)
    
    trainer=Trainer(model=model,args=args,train_dataset=train_ds,eval_dataset=val_ds,callbacks=[ProgressCB(tot_steps)])
    
    print("\n"+"="*70+"\n   TRAINING\n"+"="*70)
    t0=time.time(); trainer.train(); train_min=(time.time()-t0)/60
    
    print("\n[SAVE] Saving model...")
    trainer.save_model(str(CHECKPOINTS_DIR/"final")); proc.save_pretrained(str(CHECKPOINTS_DIR/"final"))
    
    print("\n"+"="*70+"\n   FINAL EVAL\n"+"="*70)
    final_acc,final_preds=evaluate(model,proc,test_df,desc="Final")
    
    print("\n"+"="*70+"\n   RESULTS\n"+"="*70)
    print(f"  Baseline: {base_acc:.2%}")
    print(f"  Final:    {final_acc:.2%}")
    print(f"  Improve:  +{final_acc-base_acc:.2%}")
    print(f"  Time:     {train_min:.1f} min")
    print_gpu("Final: ")
    
    print("\nFinal samples (10):")
    for p in final_preds[:10]:
        s="OK" if p["ok"] else "X "
        print(f"  [{s}] {p['true']:3} -> {p['pred']:3} | {p['path'].split('/')[-1][:25]}")
    
    metrics={"baseline":base_acc,"final":final_acc,"improve":final_acc-base_acc,"time_min":train_min,"train_n":len(train_df),"test_n":len(test_df)}
    with open(METRICS_DIR/"finetuned_binary.json","w") as f: json.dump(metrics,f,indent=2)
    pd.DataFrame(final_preds).to_csv(METRICS_DIR/"finetuned_preds.csv",index=False)
    print(f"\n[DONE] Saved to {METRICS_DIR}")

if __name__=="__main__": main()
