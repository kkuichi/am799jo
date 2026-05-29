"""
Baseline models for comparison with transformers.
Bachelor's thesis: Využitie transformerov v detekcii toxicity na sociálnych sieťach

Three classical baseline models on the same dataset (Jigsaw Toxic Comment Classification):
  1. Logistic Regression + TF-IDF
  2. Linear SVM + TF-IDF
  3. Multinomial Naive Bayes + TF-IDF

The train/validation split is identical to the transformer experiments (SEED=42), so the results are directly comparable.

Requirements:
    pip install scikit-learn pandas numpy
"""

import os
import time
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────

TRAIN_CSV  = "train.csv"
OUTPUT_DIR = "./baseline_models"
SEED       = 42       # Same seed as in the transformer experiments.
VAL_SIZE   = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 2. Data Loading (same as for transformers)
# ─────────────────────────────────────────────

print("="*60)
print("DATA LOADING")
print("="*60)

df = pd.read_csv(TRAIN_CSV)
toxic_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
df["label"] = (df[toxic_cols].sum(axis=1) > 0).astype(int)
df = df[["comment_text", "label"]].dropna()

n_total = len(df)
n_toxic = df["label"].sum()
print(f"\nDataset: {n_total} examples ({n_toxic} toxic, {100*n_toxic/n_total:.1f}%)")

train_df, val_df = train_test_split(
    df, test_size=VAL_SIZE, random_state=SEED, stratify=df["label"],
)
print(f"Training examples:     {len(train_df)}")
print(f"Validation examples: {len(val_df)}")

X_train_text = train_df["comment_text"].values
X_val_text   = val_df["comment_text"].values
y_train      = train_df["label"].values
y_val        = val_df["label"].values

# ─────────────────────────────────────────────
# 3. TF-IDF Vectorization
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("TF-IDF VECTORIZATION")
print("="*60)

# TF-IDF with appropriate parameters:
# - max_features: limit the vocabulary to the top 50k terms
# - ngram_range: use unigrams and bigrams
# - min_df=2: terms must appear in at least 2 documents
# - sublinear_tf: logarithmic term-frequency normalization
# - lowercase=True: convert text to lowercase, similarly to uncased BERT
vectorizer = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    lowercase=True,
    strip_accents="unicode",
)

t0 = time.time()
X_train = vectorizer.fit_transform(X_train_text)
X_val   = vectorizer.transform(X_val_text)
vec_time = time.time() - t0

print(f"\nVocabulary size: {len(vectorizer.vocabulary_)} words/bigrams")
print(f"Feature matrix shape: {X_train.shape}")
print(f"Vectorization time: {vec_time:.1f} sec")
print(f"Matrix density: {X_train.nnz / (X_train.shape[0] * X_train.shape[1]) * 100:.3f}%")

# Save the vectorizer
with open(os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

# ─────────────────────────────────────────────
# 4. Evaluation Function
# ─────────────────────────────────────────────

def evaluate_model(name: str, model, X_val, y_val, has_proba=True):
    """Evaluate a model and print all metrics."""
    print(f"\n{'─'*60}")
    print(f"RESULTS: {name}")
    print(f"{'─'*60}")

    t0 = time.time()
    y_pred = model.predict(X_val)
    pred_time = (time.time() - t0) * 1000 / len(y_val)  # ms per example

    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred, average="binary")

    if has_proba:
        y_prob = model.predict_proba(X_val)[:, 1]
        auc    = roc_auc_score(y_val, y_prob)
    else:
        # For LinearSVC, use the distance to the hyperplane.
        y_score = model.decision_function(X_val)
        auc     = roc_auc_score(y_val, y_score)

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Inference:  {pred_time:.3f} ms per example")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["Non-toxic", "Toxic"]))

    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    return {
        "name": name,
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "pred_time_ms": pred_time,
        "tn": int(cm[0,0]), "fp": int(cm[0,1]),
        "fn": int(cm[1,0]), "tp": int(cm[1,1]),
    }


results = []

# ─────────────────────────────────────────────
# 5. Model 1 — Logistic Regression + TF-IDF
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("TRAINING: Logistic Regression + TF-IDF")
print("="*60)

# class_weight='balanced' helps with the imbalanced dataset.
log_reg = LogisticRegression(
    max_iter=1000,
    C=4.0,                    # tuned for this dataset
    class_weight="balanced",
    solver="liblinear",
    random_state=SEED,
)

t0 = time.time()
log_reg.fit(X_train, y_train)
train_time = time.time() - t0
print(f"Training time: {train_time:.1f} sec")

result = evaluate_model("Logistic Regression + TF-IDF", log_reg, X_val, y_val)
result["train_time_sec"] = train_time
results.append(result)

# Save the model
with open(os.path.join(OUTPUT_DIR, "logistic_regression.pkl"), "wb") as f:
    pickle.dump(log_reg, f)

# ─────────────────────────────────────────────
# 6. Model 2 — Linear SVM + TF-IDF
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("TRAINING: Linear SVM + TF-IDF")
print("="*60)

svm = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=SEED,
    max_iter=2000,
)

t0 = time.time()
svm.fit(X_train, y_train)
train_time = time.time() - t0
print(f"Training time: {train_time:.1f} sec")

result = evaluate_model("Linear SVM + TF-IDF", svm, X_val, y_val, has_proba=False)
result["train_time_sec"] = train_time
results.append(result)

with open(os.path.join(OUTPUT_DIR, "linear_svm.pkl"), "wb") as f:
    pickle.dump(svm, f)

# ─────────────────────────────────────────────
# 7. Model 3 — Multinomial Naive Bayes + TF-IDF
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("TRAINING: Multinomial Naive Bayes + TF-IDF")
print("="*60)

# Naive Bayes does not natively support class_weight,
# but it usually performs reasonably well on imbalanced data.
nb = MultinomialNB(alpha=0.1)

t0 = time.time()
nb.fit(X_train, y_train)
train_time = time.time() - t0
print(f"Training time: {train_time:.1f} sec")

result = evaluate_model("Multinomial Naive Bayes + TF-IDF", nb, X_val, y_val)
result["train_time_sec"] = train_time
results.append(result)

with open(os.path.join(OUTPUT_DIR, "naive_bayes.pkl"), "wb") as f:
    pickle.dump(nb, f)

# ─────────────────────────────────────────────
# 8. Final Comparison Table of Baseline Models
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("BASELINE MODEL COMPARISON")
print("="*60)

results_df = pd.DataFrame(results)
print("\n", results_df[["name", "accuracy", "f1", "auc", "pred_time_ms", "train_time_sec"]].to_string(index=False))

results_df.to_csv(os.path.join(OUTPUT_DIR, "baseline_results.csv"), index=False)
print(f"\nResults saved to {OUTPUT_DIR}/baseline_results.csv")

# ─────────────────────────────────────────────
# 9. Comparison with Transformers
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("SUMMARY: BASELINE vs TRANSFORMERS")
print("="*60)

transformer_results = [
    ("BERT-base",      110.0, 0.8427, 0.9864),
    ("RoBERTa-base",   125.0, 0.8370, 0.9862),
    ("DistilBERT",      67.0, 0.8327, 0.9841),
    ("ALBERT-base-v2",  11.7, 0.8288, 0.9842),
]

print(f"\n{'Model':<40} {'Parameters':>12} {'F1':>8} {'AUC':>8}")
print("─" * 70)
for r in results:
    print(f"{r['name']:<40} {'~0.05M':>12} {r['f1']:>8.4f} {r['auc']:>8.4f}")
print("─" * 70)
for name, params, f1, auc in transformer_results:
    print(f"{name:<40} {params:>11.1f}M {f1:>8.4f} {auc:>8.4f}")

# ─────────────────────────────────────────────
# 10. Prediction Examples on the Same Texts
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("PREDICTION EXAMPLES")
print("="*60)

examples = [
    "You are a wonderful person and I appreciate your help!",
    "You are an idiot and I hope you die.",
    "I disagree with your opinion on this matter.",
    "Go kill yourself you worthless piece of garbage.",
]

models = [
    ("LogReg", log_reg, True),
    ("SVM",    svm,     False),
    ("NB",     nb,      True),
]

X_examples = vectorizer.transform(examples)

for i, text in enumerate(examples):
    print(f"\n  Text: {text}")
    for name, model, has_proba in models:
        pred = model.predict(X_examples[i])[0]
        if has_proba:
            prob = model.predict_proba(X_examples[i])[0, 1]
            print(f"    {name:<8}: {'TOXIC' if pred == 1 else 'NOT TOXIC':<10} (p={prob:.3f})")
        else:
            score = model.decision_function(X_examples[i])[0]
            print(f"    {name:<8}: {'TOXIC' if pred == 1 else 'NOT TOXIC':<10} (score={score:.3f})")