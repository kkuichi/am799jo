"""
Inference speed benchmark for four transformer models.

Bachelor's thesis:
Využitie transformerov v detekcii toxicity na sociálnych sieťach

This script measures inference latency and throughput for the fine-tuned
BERT, RoBERTa, DistilBERT, and ALBERT models.

Run this script from the project root directory.

Requirements:
    pip install transformers torch numpy
"""

import time

import numpy as np
import torch
from transformers import (
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
N_WARMUP = 50
N_RUNS = 200

MODELS = [
    ("BERT", "./bert_toxicity_model", BertTokenizer, BertForSequenceClassification),
    ("RoBERTa", "./roberta_toxicity_model", RobertaTokenizer, RobertaForSequenceClassification),
    ("DistilBERT", "./distilbert_toxicity_model", DistilBertTokenizer, DistilBertForSequenceClassification),
    ("ALBERT", "./albert_toxicity_model", AlbertTokenizer, AlbertForSequenceClassification),
]


# --------------------------------------------------
# 2. Benchmark Input Text
# --------------------------------------------------

# Test sample with a typical length similar to comments in the Jigsaw dataset.
TEST_TEXT = (
    "This is a sample comment used for inference speed benchmarking. "
    "It contains a typical length of text found in online discussions."
)


# --------------------------------------------------
# 3. Benchmark Execution
# --------------------------------------------------

print(f"Device: {DEVICE}")
print(f"Warm-up runs: {N_WARMUP}")
print(f"Measured runs: {N_RUNS}")
print("=" * 60)

results = []

for model_name, model_path, TokenizerClass, ModelClass in MODELS:
    print(f"\nLoading {model_name}...")

    tokenizer = TokenizerClass.from_pretrained(model_path)
    model = ModelClass.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()

    inputs = tokenizer(
        TEST_TEXT,
        return_tensors="pt",
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
    )
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
        if key in ["input_ids", "attention_mask"]
    }

    # Warm-up phase
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(**inputs)

    # Measurement phase
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    times = []

    with torch.no_grad():
        for _ in range(N_RUNS):
            start = time.perf_counter()
            _ = model(**inputs)
            torch.cuda.synchronize() if DEVICE.type == "cuda" else None
            end = time.perf_counter()
            times.append((end - start) * 1000)

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    throughput = 1000 / mean_ms

    results.append((model_name, mean_ms, std_ms, throughput))
    print(f"  Average latency: {mean_ms:.2f} ms ± {std_ms:.2f} ms")
    print(f"  Throughput: {throughput:.0f} comments/sec")


# --------------------------------------------------
# 4. Summary Table
# --------------------------------------------------

print("\n" + "=" * 60)
print(f"{'Model':<12} {'Inference (ms)':<18} {'Throughput (/s)'}")
print("=" * 60)

for model_name, mean_ms, std_ms, throughput in results:
    print(f"{model_name:<12} {mean_ms:.2f} ± {std_ms:.2f} ms    {throughput:.0f}/s")
