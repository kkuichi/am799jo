"""
Error analysis for the discussion chapter of the bachelor thesis.
Bachelor's thesis: Využitie transformerov v detekcii toxicity na sociálnych sieťach

This script:
  1. Runs the selected model on the Jigsaw validation set
  2. Finds all errors (FN and FP)
  3. Saves them to CSV files sorted by confidence
  4. Prints the most relevant error categories for analysis

Usage:
    python error_analysis.py
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MODEL_DIR  = "./bert_toxicity_model"   # analyze the best model (BERT)
JIGSAW_CSV = "train.csv"
OUTPUT_DIR = "./error_analysis"
MAX_LEN    = 256
BATCH_SIZE = 32
SEED       = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
        }


# ─────────────────────────────────────────────
# Data and Model Loading
# ─────────────────────────────────────────────

print("="*70)
print("ERROR ANALYSIS")
print("="*70)

print("\nLoading Jigsaw validation...")
df = pd.read_csv(JIGSAW_CSV)
toxic_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
df["label"] = (df[toxic_cols].sum(axis=1) > 0).astype(int)
df = df[["comment_text"] + toxic_cols + ["label"]].dropna()

# Same split as during training.
_, val_df = train_test_split(
    df, test_size=0.1, random_state=SEED, stratify=df["label"],
)
val_df = val_df.reset_index(drop=True)
print(f"  Loaded: {len(val_df)} examples")

print(f"\nLoading model from {MODEL_DIR}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model     = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

print("\nRunning inference on the validation set...")
dataset = InferenceDataset(val_df["comment_text"], tokenizer, MAX_LEN)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

all_probs = []
with torch.no_grad():
    for batch in tqdm(loader):
        with autocast():
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                token_type_ids=batch["token_type_ids"].to(DEVICE),
            )
        probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().float().numpy()
        all_probs.extend(probs)

val_df["pred_prob"]  = all_probs
val_df["pred_label"] = (val_df["pred_prob"] >= 0.5).astype(int)
val_df["correct"]    = val_df["pred_label"] == val_df["label"]

# ─────────────────────────────────────────────
# Error Categorization
# ─────────────────────────────────────────────

# False Negatives — the model predicted "non-toxic", but the true label is toxic
false_negatives = val_df[(val_df["label"] == 1) & (val_df["pred_label"] == 0)].copy()
false_negatives = false_negatives.sort_values("pred_prob", ascending=True)  # most confident FN cases first

# False Positives — the model predicted "toxic", but the true label is non-toxic
false_positives = val_df[(val_df["label"] == 0) & (val_df["pred_label"] == 1)].copy()
false_positives = false_positives.sort_values("pred_prob", ascending=False)  # most confident FP cases first

# Borderline cases — the model is uncertain
borderline = val_df[(val_df["pred_prob"] >= 0.4) & (val_df["pred_prob"] <= 0.6)].copy()
borderline = borderline.sort_values("pred_prob")

print(f"\n{'='*70}")
print("ERROR SUMMARY")
print(f"{'='*70}")
print(f"  Total validation examples:     {len(val_df)}")
print(f"  Correct:               {val_df['correct'].sum()} ({100*val_df['correct'].mean():.2f}%)")
print(f"  Total errors:            {(~val_df['correct']).sum()} ({100*(~val_df['correct']).mean():.2f}%)")
print(f"  False Negatives (FN):    {len(false_negatives)} — missed toxic comments")
print(f"  False Positives (FP):    {len(false_positives)} — false alarms")
print(f"  Borderline (0.4-0.6):    {len(borderline)} — model is uncertain")

# ─────────────────────────────────────────────
# Save All Errors to CSV for Manual Analysis
# ─────────────────────────────────────────────

# Save toxicity subtypes to support manual analysis.
cols_to_save = ["comment_text", "label", "pred_label", "pred_prob"] + toxic_cols

false_negatives[cols_to_save].to_csv(
    os.path.join(OUTPUT_DIR, "false_negatives.csv"), index=False
)
false_positives[cols_to_save].to_csv(
    os.path.join(OUTPUT_DIR, "false_positives.csv"), index=False
)
borderline[cols_to_save].to_csv(
    os.path.join(OUTPUT_DIR, "borderline_cases.csv"), index=False
)

print(f"\nFiles saved to {OUTPUT_DIR}/")
print(f"  • false_negatives.csv  — sorted with the most confident errors first")
print(f"  • false_positives.csv  — same sorting")
print(f"  • borderline_cases.csv — borderline cases")

# ─────────────────────────────────────────────
# Helper Function for Printing Examples
# ─────────────────────────────────────────────

def print_example(row, idx, max_chars=300):
    """Print one example with metadata."""
    text = row["comment_text"]
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    # Identify active toxicity subtypes.
    active_subtypes = [s for s in toxic_cols if row[s] == 1]
    subtypes_str = ", ".join(active_subtypes) if active_subtypes else "—"

    print(f"\n  [{idx}] p_toxic = {row['pred_prob']:.3f}")
    print(f"      True: {row['label']} | Pred: {row['pred_label']}")
    if active_subtypes:
        print(f"      Toxicity subtypes: {subtypes_str}")
    print(f"      Text: {text}")


# ─────────────────────────────────────────────
# Print Error Categories
# ─────────────────────────────────────────────

print(f"\n\n{'='*70}")
print("TOP-15 MOST CONFIDENT FALSE NEGATIVES")
print('(the model confidently predicted "non-toxic", but the true label is toxic)')
print(f"{'='*70}")
for i, (_, row) in enumerate(false_negatives.head(15).iterrows(), 1):
    print_example(row, i)

print(f"\n\n{'='*70}")
print("TOP-15 MOST CONFIDENT FALSE POSITIVES")
print('(the model confidently predicted "toxic", but the true label is non-toxic)')
print(f"{'='*70}")
for i, (_, row) in enumerate(false_positives.head(15).iterrows(), 1):
    print_example(row, i)

print(f"\n\n{'='*70}")
print("BORDERLINE CASES (model is uncertain, 0.4 < p < 0.6)")
print(f"{'='*70}")
# Take 10 random borderline cases.
borderline_sample = borderline.sample(min(10, len(borderline)), random_state=SEED)
for i, (_, row) in enumerate(borderline_sample.iterrows(), 1):
    print_example(row, i)

# ─────────────────────────────────────────────
# Statistics by Toxicity Subtype
# ─────────────────────────────────────────────

print(f"\n\n{'='*70}")
print("FN STATISTICS BY TOXICITY SUBTYPE")
print(f"{'='*70}")
print("\nSubtype:          Total / Missed / Miss rate")
for subtype in toxic_cols:
    total = (val_df[subtype] == 1).sum()
    missed = ((val_df[subtype] == 1) & (val_df["pred_label"] == 0)).sum()
    pct = 100 * missed / total if total > 0 else 0
    print(f"  {subtype:<18} {total:>6}   /   {missed:>5}    /  {pct:>5.1f}%")

# ─────────────────────────────────────────────
# Model Confidence Distribution on Errors
# ─────────────────────────────────────────────

print(f"\n\n{'='*70}")
print("MODEL CONFIDENCE DISTRIBUTION ON ERRORS")
print(f"{'='*70}")

print('\nFalse Negatives (the model should have predicted "toxic"):')
print(f"  Very confident (p < 0.1):     {(false_negatives['pred_prob'] < 0.1).sum():>5} cases")
print(f"  Confident (0.1 ≤ p < 0.3):     {((false_negatives['pred_prob'] >= 0.1) & (false_negatives['pred_prob'] < 0.3)).sum():>5} cases")
print(f"  Uncertain (0.3 ≤ p < 0.5): {((false_negatives['pred_prob'] >= 0.3) & (false_negatives['pred_prob'] < 0.5)).sum():>5} cases")

print('\nFalse Positives (the model should have predicted "non-toxic"):')
print(f"  Uncertain (0.5 ≤ p < 0.7): {((false_positives['pred_prob'] >= 0.5) & (false_positives['pred_prob'] < 0.7)).sum():>5} cases")
print(f"  Confident (0.7 ≤ p < 0.9):     {((false_positives['pred_prob'] >= 0.7) & (false_positives['pred_prob'] < 0.9)).sum():>5} cases")
print(f"  Very confident (p ≥ 0.9):     {(false_positives['pred_prob'] >= 0.9).sum():>5} cases")

print("\n" + "="*70)
print("DONE")
print("="*70)
print(f"""
Next steps:
  1. Open {OUTPUT_DIR}/false_negatives.csv in Excel/Numbers
  2. Read the first ~30-50 examples, which are the most confident errors.
  3. Manually categorize them: sarcasm, quotation, slang, typos, etc.
  4. Repeat the same process for false_positives.csv.
  5. Use these observations in the "Discussion / Error Analysis" chapter.
""")