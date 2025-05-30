# Importovanie knižníc pre webový server a NLP modely
from flask import Flask, render_template, request, send_file
from transformers import pipeline, RobertaTokenizer, RobertaForSequenceClassification
from tiny_toxic_detector import load_tiny_model, predict_tiny
import torch
import torch.nn.functional as F
import time
import csv
import io

# Inicializácia Flask aplikácie
app = Flask(__name__)
history = []  # História požiadaviek na analýzu toxicity

# Načítanie troch rôznych modelov pre detekciu toxicity
# 1. Tiny model optimalizovaný na rýchlosť
tiny_model, tiny_tokenizer, tiny_device = load_tiny_model()

# 2. Model RoBERTa pre klasifikáciu toxicity
roberta_tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
roberta_model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier').to("cpu")
roberta_model.eval()

# 3. Predtrénovaný BERT model cez transformers pipeline
bert_pipeline = pipeline("text-classification", model="unitary/toxic-bert")

# Funkcia na predikciu toxicity pomocou RoBERTa
def predict_roberta(text):
    start = time.perf_counter()
    inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
    latency = (time.perf_counter() - start) * 1000
    return round(probs[0][1].item(), 3), round(latency, 2)  # Pravdepodobnosť toxicity a latencia

# Funkcia na predikciu toxicity pomocou BERT pipeline
def predict_bert(text):
    start = time.perf_counter()
    try:
        out = bert_pipeline(text)[0]
        latency = (time.perf_counter() - start) * 1000
        return round(out["score"], 3), round(latency, 2)
    except Exception as e:
        return f"Error: {str(e)}", 0

# Hlavná stránka aplikácie – spracovanie GET/POST požiadaviek
@app.route("/", methods=["GET", "POST"])
def index():
    global history
    result = {}
    action = request.form.get("action", "check")
    text = request.form.get("text", "").strip()

    # Vyčistenie histórie
    if action == "clear":
        history = []

    # Export histórie do CSV
    elif action == "download":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Input", "Tiny", "RoBERTa", "BERT"])
        for row in history:
            writer.writerow([row["text"], row["tiny"], row["roberta"], row["bert"]])
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name="toxicity_history.csv"
        )

    # Výpočet toxicity pri zadaní textu
    elif action == "check" and text:
        tiny_score, tiny_latency = predict_tiny(text, tiny_model, tiny_tokenizer, tiny_device)
        roberta_score, roberta_latency = predict_roberta(text)
        bert_score, bert_latency = predict_bert(text)

        # Uloženie výsledkov jednotlivých modelov
        result = {
            "Tiny-Toxic-Detector": {
                "score": tiny_score,
                "latency": tiny_latency
            },
            "RoBERTa (s-nlp)": {
                "score": roberta_score,
                "latency": roberta_latency
            },
            "BERT (unitary)": {
                "score": bert_score,
                "latency": bert_latency
            }
        }

        # Pridanie výsledkov do histórie
        history.append({
            "text": text,
            "tiny": tiny_score,
            "roberta": roberta_score,
            "bert": bert_score
        })

    # Vykreslenie hlavnej stránky s výsledkami
    return render_template("index.html", result=result, history=history)

# Spustenie Flask aplikácie v debug móde
if __name__ == "__main__":
    app.run(debug=True)
