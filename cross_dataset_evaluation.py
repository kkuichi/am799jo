"""
Cross-dataset evaluation of toxicity detection models.
Bachelor's thesis: Využitie transformerov v detekcii toxicity na sociálnych sieťach

This script evaluates seven trained models (four transformer models and three baseline models) on three datasets:
  1. Jigsaw validation (in-distribution)
  2. Civil Comments     (out-of-distribution, news comments)
  3. TweetEval Offensive (out-of-distribution, tweets)

Objective: evaluate how well the models generalize outside the training domain.

Requirements:
    pip install transformers torch scikit-learn pandas tqdm datasets
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
)

from datasets import load_dataset

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

JIGSAW_CSV   = "train.csv"
BASELINE_DIR = "./baseline_models"
TRANSFORMER_DIRS = {
    "BERT":       ("./bert_toxicity_model",       BertTokenizer,       BertForSequenceClassification,       True),
    "RoBERTa":    ("./roberta_toxicity_model",    RobertaTokenizer,    RobertaForSequenceClassification,    False),
    "DistilBERT": ("./distilbert_toxicity_model", DistilBertTokenizer, DistilBertForSequenceClassification, False),
    "ALBERT":     ("./albert_toxicity_model",     AlbertTokenizer,     AlbertForSequenceClassification,     False),
}
# Tuple format: (directory, tokenizer, model, uses_token_type_ids)

MAX_LEN     = 256
BATCH_SIZE  = 32
N_EXTERNAL  = 20_000
SEED        = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {DEVICE}\n")

# ─────────────────────────────────────────────
# Inference Dataset
# ─────────────────────────────────────────────

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, use_token_type_ids):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_tti = use_token_type_ids

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
        out = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.use_tti and "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return out


# ─────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────

def load_jigsaw_val():
    """Return (texts, labels) for the Jigsaw validation split used during training."""
    print("Loading Jigsaw validation...")
    df = pd.read_csv(JIGSAW_CSV)
    toxic_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    df["label"] = (df[toxic_cols].sum(axis=1) > 0).astype(int)
    df = df[["comment_text", "label"]].dropna()

    # Same split as during training.
    _, val_df = train_test_split(
        df, test_size=0.1, random_state=SEED, stratify=df["label"],
    )
    texts  = val_df["comment_text"].tolist()
    labels = val_df["label"].tolist()
    print(f"  Loaded: {len(texts)} examples ({sum(labels)} toxic, {100*sum(labels)/len(labels):.1f}%)")
    return texts, labels


def load_civil_comments(n=N_EXTERNAL):
    """Civil Comments — comments from news websites."""
    print(f"\nLoading Civil Comments (up to {n} examples)...")
    ds = load_dataset("civil_comments", split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))

    texts  = [str(t) for t in ds["text"]]
    # Toxicity is a float from 0 to 1; binarize using the 0.5 threshold, as in Jigsaw.
    labels = [1 if t >= 0.5 else 0 for t in ds["toxicity"]]
    print(f"  Loaded: {len(texts)} examples ({sum(labels)} toxic, {100*sum(labels)/len(labels):.1f}%)")
    return texts, labels


def load_tweet_eval(n=N_EXTERNAL):
    """TweetEval Offensive — tweets labeled as offensive/normal."""
    print(f"\nLoading TweetEval Offensive (up to {n} examples)...")
    # Combine all splits to use the maximum available number of samples.
    parts = []
    for split in ["train", "validation", "test"]:
        parts.append(load_dataset("tweet_eval", "offensive", split=split))
    from datasets import concatenate_datasets
    ds = concatenate_datasets(parts)
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))

    texts  = [str(t) for t in ds["text"]]
    labels = list(ds["label"])
    print(f"  Loaded: {len(texts)} examples ({sum(labels)} toxic, {100*sum(labels)/len(labels):.1f}%)")
    return texts, labels


# ─────────────────────────────────────────────
# Transformer Inference
# ─────────────────────────────────────────────

@torch.no_grad()
def infer_transformer(model, loader, device, use_tti):
    model.eval()
    all_probs = []

    for batch in tqdm(loader, desc="  Inference", leave=False):
        inputs = {
            "input_ids":      batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        if use_tti and "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"].to(device)

        with autocast():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().float().numpy()
        all_probs.extend(probs)

    return np.array(all_probs)


def evaluate_transformer(name, model_dir, tokenizer_cls, model_cls, use_tti, texts, labels):
    print(f"\n  ▸ {name}")
    tokenizer = tokenizer_cls.from_pretrained(model_dir)
    model     = model_cls.from_pretrained(model_dir).to(DEVICE)

    dataset = InferenceDataset(texts, tokenizer, MAX_LEN, use_tti)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    probs = infer_transformer(model, loader, DEVICE, use_tti)
    preds = (probs >= 0.5).astype(int)

    # Free memory.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return compute_metrics(name, np.array(labels), preds, probs)


# ─────────────────────────────────────────────
# Baseline Model Inference
# ─────────────────────────────────────────────

def evaluate_baseline(name, model, vectorizer, texts, labels, has_proba=True):
    print(f"\n  ▸ {name}")
    X = vectorizer.transform(texts)
    preds = model.predict(X)

    if has_proba:
        probs = model.predict_proba(X)[:, 1]
    else:
        # For LinearSVC, use decision_function for AUC.
        probs = model.decision_function(X)

    return compute_metrics(name, np.array(labels), preds, probs)


# ─────────────────────────────────────────────
# Metric Computation
# ─────────────────────────────────────────────

def compute_metrics(name, y_true, y_pred, y_score):
    return {
        "model":     name,
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, average="binary", zero_division=0),
        "auc":       roc_auc_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
    }


# ─────────────────────────────────────────────
# Main Evaluation Function
# ─────────────────────────────────────────────

def evaluate_all_models_on(dataset_name, texts, labels):
    """Evaluate all seven models on one dataset."""
    print(f"\n{'='*70}")
    print(f"EVALUATION ON: {dataset_name}")
    print(f"{'='*70}")

    results = []

    # Four transformer models
    for name, (model_dir, tok_cls, model_cls, use_tti) in TRANSFORMER_DIRS.items():
        if not os.path.exists(model_dir):
            print(f"\n  ⚠ Model {name} not found in {model_dir}, skipping")
            continue
        results.append(evaluate_transformer(name, model_dir, tok_cls, model_cls, use_tti, texts, labels))

    # Three baseline models
    print(f"\n  Loading baseline models...")
    with open(os.path.join(BASELINE_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(BASELINE_DIR, "logistic_regression.pkl"), "rb") as f:
        log_reg = pickle.load(f)
    with open(os.path.join(BASELINE_DIR, "linear_svm.pkl"), "rb") as f:
        svm = pickle.load(f)
    with open(os.path.join(BASELINE_DIR, "naive_bayes.pkl"), "rb") as f:
        nb = pickle.load(f)

    results.append(evaluate_baseline("LogReg+TFIDF", log_reg, vectorizer, texts, labels, has_proba=True))
    results.append(evaluate_baseline("SVM+TFIDF",    svm,     vectorizer, texts, labels, has_proba=False))
    results.append(evaluate_baseline("NaiveBayes",   nb,      vectorizer, texts, labels, has_proba=True))

    return results


# ─────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load all datasets ──
    print("="*70)
    print("DATASET LOADING")
    print("="*70 + "\n")

    datasets_to_test = {
        "Jigsaw (trained on)": load_jigsaw_val(),
        "Civil Comments":      load_civil_comments(),
        "TweetEval Offensive": load_tweet_eval(),
    }

    # ── Evaluation ──
    all_results = {}
    for ds_name, (texts, labels) in datasets_to_test.items():
        all_results[ds_name] = evaluate_all_models_on(ds_name, texts, labels)

    # ── Build final results table ──
    print("\n" + "="*100)
    print("FINAL TABLE — ALL MODELS x ALL DATASETS")
    print("="*100)

    # Get the list of models from the first dataset.
    model_names = [r["model"] for r in all_results[list(all_results.keys())[0]]]
    metric_names = ["accuracy", "f1", "auc", "precision", "recall"]

    # Print each metric separately for better console readability.
    for metric in metric_names:
        print(f"\n── {metric.upper()} ──")
        # Header
        header = f"{'Model':<18}"
        for ds_name in datasets_to_test.keys():
            header += f"{ds_name:>22}"
        print(header)
        print("─" * len(header))

        # Rows
        for model_name in model_names:
            row = f"{model_name:<18}"
            for ds_name in datasets_to_test.keys():
                value = next(r[metric] for r in all_results[ds_name] if r["model"] == model_name)
                row += f"{value:>22.4f}"
            print(row)

    # ── Save results to CSV for use in the thesis ──
    rows = []
    for ds_name, results in all_results.items():
        for r in results:
            rows.append({"dataset": ds_name, **r})

    df_results = pd.DataFrame(rows)
    df_results.to_csv("./cross_dataset_results.csv", index=False)
    print(f"\n\nResults saved to cross_dataset_results.csv")

    # ── Additional F1-score pivot table ──
    print("\n" + "="*70)
    print("SUMMARY TABLE: F1-SCORE")
    print("="*70)
    pivot_f1 = df_results.pivot(index="model", columns="dataset", values="f1")
    # Sort columns in the required order.
    pivot_f1 = pivot_f1[list(datasets_to_test.keys())]
    # Sort rows: transformer models first, then baseline models.
    model_order = list(TRANSFORMER_DIRS.keys()) + ["LogReg+TFIDF", "SVM+TFIDF", "NaiveBayes"]
    pivot_f1 = pivot_f1.reindex([m for m in model_order if m in pivot_f1.index])
    print(pivot_f1.round(4).to_string())

    # F1-score drop for each model.
    print("\n" + "="*70)
    print("F1-SCORE DROP ON OUT-OF-DISTRIBUTION DATA")
    print("="*70)
    jigsaw_col = "Jigsaw (trained on)"
    for ds_name in datasets_to_test.keys():
        if ds_name == jigsaw_col:
            continue
        print(f"\n{jigsaw_col} → {ds_name}:")
        for model in pivot_f1.index:
            jigsaw_f1 = pivot_f1.loc[model, jigsaw_col]
            ood_f1    = pivot_f1.loc[model, ds_name]
            drop      = jigsaw_f1 - ood_f1
            drop_pct  = (drop / jigsaw_f1) * 100
            print(f"  {model:<18} {jigsaw_f1:.4f} → {ood_f1:.4f}  (Δ = -{drop:.4f}, -{drop_pct:.1f}%)")

    print("\nDone!")