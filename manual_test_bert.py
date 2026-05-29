import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# =========================
# PATH К ТВОЕЙ МОДЕЛИ
# =========================
MODEL_PATH = "./results"   # папка где сохранилась модель

# =========================
# LOAD MODEL + TOKENIZER
# =========================

print("Loading model...")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# =========================
# DEVICE (GPU если есть)
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =========================
# TEST TEXTS
# =========================

texts = [
    "You are an amazing person!",
    "I hate you, you're stupid",
    "Have a nice day :)",
    "Go kill yourself",
    "This is really helpful, thanks!",
    "You're the worst human being ever",
    "I love this project",
    "Shut up idiot"
]

# =========================
# PREDICTION FUNCTION
# =========================

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        probs = F.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    label = "TOXIC" if pred_class.item() == 1 else "NON-TOXIC"

    return label, confidence.item()

# =========================
# RUN TESTS
# =========================

print("\n===== PREDICTIONS =====\n")

for text in texts:
    label, conf = predict(text)
    print(text)
    print(f"→ {label} (confidence: {conf:.3f})")
    print("-" * 50)