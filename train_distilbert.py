"""
DistilBERT fine-tuning for toxicity detection
(Jigsaw Toxic Comment Classification Challenge)

Bachelor's thesis:
Využitie transformerov v detekcii toxicity na sociálnych sieťach

Third experiment: DistilBERT is a distilled, smaller version of BERT.
- 6 layers instead of 12
- 66M parameters instead of 110M (-40%)
- Training and inference are approximately 2x faster
- Quality is approximately 97% of the original BERT model

The hyperparameters are identical to BERT and RoBERTa for a fair comparison.

Requirements:
    pip install transformers torch scikit-learn pandas tqdm
"""

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

class Config:
    TRAIN_CSV    = "train.csv"
    MODEL_NAME   = "distilbert-base-uncased"
    OUTPUT_DIR   = "./distilbert_toxicity_model"

    MAX_LEN       = 256
    BATCH_SIZE    = 16        # Same as BERT/RoBERTa for a fair comparison.
    EPOCHS        = 3
    LEARNING_RATE = 2e-5
    WARMUP_RATIO  = 0.1
    WEIGHT_DECAY  = 0.01
    VAL_SIZE      = 0.1
    SEED          = 42

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 2. Dataset
# --------------------------------------------------
# DistilBERT, like RoBERTa, does not use token_type_ids.

class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts.reset_index(drop=True)
        self.labels    = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# --------------------------------------------------
# 3. Training and evaluation
# --------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="binary")
    return avg_loss, acc, f1


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            total_loss += outputs.loss.item()
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().float().numpy()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="binary")
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, f1, auc, all_preds, all_labels


def predict(text: str, model, tokenizer, device, max_len=256) -> dict:
    """Make a prediction for one comment."""
    model.eval()
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        with autocast():
            outputs = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            )
    probs = torch.softmax(outputs.logits, dim=1)[0].cpu().float().numpy()
    label = int(np.argmax(probs))
    return {
        "label":          "TOXIC" if label == 1 else "NOT TOXIC",
        "toxic_prob":     float(probs[1]),
        "not_toxic_prob": float(probs[0]),
    }


# --------------------------------------------------
# Inference speed benchmark for the practical evaluation section
# --------------------------------------------------

def benchmark_inference_speed(model, tokenizer, device, n_samples=500, max_len=256):
    """Measure the average inference time for one comment."""
    sample_text = "This is a sample comment to measure inference speed of the model."
    encoding = tokenizer(
        sample_text, max_length=max_len, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    model.eval()

    # GPU warm-up. The first calls are usually slower.
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_samples):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    avg_ms = (elapsed / n_samples) * 1000
    print(f"\nAverage inference time: {avg_ms:.2f} ms per comment")
    print(f"Throughput: {n_samples/elapsed:.1f} comments/sec")
    return avg_ms


# --------------------------------------------------
# Important: on Windows, all executable code
# must be placed inside this block.
# --------------------------------------------------

if __name__ == "__main__":

    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    print(f"Model: {cfg.MODEL_NAME}")
    print(f"Device used: {cfg.DEVICE}")
    print(f"Mixed Precision (AMP): ENABLED\n")

    # Data loading
    df = pd.read_csv(cfg.TRAIN_CSV)
    toxic_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    df["label"] = (df[toxic_cols].sum(axis=1) > 0).astype(int)
    df = df[["comment_text", "label"]].dropna()

    n_total = len(df)
    n_toxic = df["label"].sum()
    print(f"Dataset loaded: {n_total} examples")
    print(f"  Toxic:       {n_toxic} ({100*n_toxic/n_total:.1f}%)")
    print(f"  Non-toxic:   {n_total - n_toxic} ({100*(n_total-n_toxic)/n_total:.1f}%)")

    train_df, val_df = train_test_split(
        df,
        test_size=cfg.VAL_SIZE,
        random_state=cfg.SEED,    # Same seed as for BERT/RoBERTa.
        stratify=df["label"],
    )
    print(f"\nTraining examples:   {len(train_df)}")
    print(f"Validation examples: {len(val_df)}")

    # Tokenizer and DataLoader
    tokenizer = DistilBertTokenizer.from_pretrained(cfg.MODEL_NAME)

    train_dataset = ToxicDataset(train_df["comment_text"], train_df["label"], tokenizer, cfg.MAX_LEN)
    val_dataset   = ToxicDataset(val_df["comment_text"],   val_df["label"],   tokenizer, cfg.MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        cfg.MODEL_NAME,
        num_labels=2,
    )
    model.to(cfg.DEVICE)

    # Parameter count, useful for the thesis comparison.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params/1e6:.1f}M")

    # Optimizer, scheduler, and AMP scaler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.LEARNING_RATE)

    total_steps  = len(train_loader) * cfg.EPOCHS
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler()

    # Training
    print("\n" + "="*60)
    print("STARTING DISTILBERT TRAINING (with AMP)")
    print("="*60)

    total_train_start = time.time()
    best_val_f1 = 0.0
    history = []

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\n-- Epoch {epoch}/{cfg.EPOCHS} --")

        epoch_start = time.time()
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg.DEVICE
        )
        val_loss, val_acc, val_f1, val_auc, val_preds, val_labels = eval_epoch(
            model, val_loader, cfg.DEVICE
        )
        epoch_time = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss":   val_loss,   "val_acc":   val_acc,   "val_f1":   val_f1,
            "val_auc":    val_auc,    "epoch_time_sec": epoch_time,
        })

        print(f"  Train — Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
        print(f"  Epoch time: {epoch_time/60:.1f} min")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(cfg.OUTPUT_DIR)
            tokenizer.save_pretrained(cfg.OUTPUT_DIR)
            print(f"  Best model saved (Val F1: {best_val_f1:.4f})")

    total_train_time = time.time() - total_train_start
    print(f"\nTotal training time: {total_train_time/60:.1f} min")

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION (best model)")
    print("="*60)

    best_model = DistilBertForSequenceClassification.from_pretrained(cfg.OUTPUT_DIR)
    best_model.to(cfg.DEVICE)

    _, val_acc, val_f1, val_auc, val_preds, val_labels = eval_epoch(
        best_model, val_loader, cfg.DEVICE
    )

    print(f"\nAccuracy:  {val_acc:.4f}")
    print(f"F1-score:  {val_f1:.4f}")
    print(f"ROC-AUC:   {val_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=["Non-toxic", "Toxic"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(val_labels, val_preds)
    print(cm)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    pd.DataFrame(history).to_csv(os.path.join(cfg.OUTPUT_DIR, "training_history.csv"), index=False)
    print(f"\nTraining history saved to {cfg.OUTPUT_DIR}/training_history.csv")

    # Inference speed benchmark for the practical evaluation section
    print("\n" + "="*60)
    print("INFERENCE SPEED BENCHMARK")
    print("="*60)
    benchmark_inference_speed(best_model, tokenizer, cfg.DEVICE)

    # Model size on disk
    model_size_mb = sum(
        os.path.getsize(os.path.join(cfg.OUTPUT_DIR, f))
        for f in os.listdir(cfg.OUTPUT_DIR)
        if f.endswith(('.bin', '.safetensors'))
    ) / (1024 * 1024)
    print(f"Model size on disk: {model_size_mb:.1f} MB")

    # Prediction examples
    print("\n" + "="*60)
    print("PREDICTION EXAMPLES")
    print("="*60)

    examples = [
        "You are a wonderful person and I appreciate your help!",
        "You are an idiot and I hope you die.",
        "I disagree with your opinion on this matter.",
        "Go kill yourself you worthless piece of garbage.",
    ]

    for text in examples:
        result = predict(text, best_model, tokenizer, cfg.DEVICE)
        print(f"\n  Text: {text[:65]}")
        print(f"  Prediction: {result['label']} (p_toxic={result['toxic_prob']:.3f})")
