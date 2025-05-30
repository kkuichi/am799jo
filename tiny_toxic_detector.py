import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
import time

# Definícia jednoduchej transformerovej architektúry na klasifikáciu textu
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        # Vstupná embedding vrstva pre prevod tokenov na vektory
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Pózičné kódovanie pre reprezentáciu poradia slov
        self.pos_encoding = nn.Parameter(torch.zeros(1, 512, embed_dim))
        # Jeden transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        # Viacnásobné vrstvy encoderov
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Lineárna vrstva pre výstup
        self.fc = nn.Linear(embed_dim, 1)
        # Sigmoid funkcia pre pravdepodobnosť toxicity
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Spojenie embeddingov a pózičného kódovania
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        # Prechod cez transformer encoder
        x = self.transformer(x)
        # Získanie priemeru po sekvencii (global average pooling)
        x = x.mean(dim=1)
        # Aplikácia lineárnej vrstvy a sigmoid výstupu
        x = self.fc(x)
        return self.sigmoid(x)

# Konfiguračná trieda pre model – definuje parametre architektúry
class TinyTransformerConfig(PretrainedConfig):
    model_type = "tiny_transformer"
    def __init__(self, vocab_size=30522, embed_dim=64, num_heads=2, ff_dim=128, num_layers=4,
                 max_position_embeddings=512, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.max_position_embeddings = max_position_embeddings

# Obalová trieda pre model s klasifikáciou sekvencie (kompatibilná s HuggingFace)
class TinyTransformerForSequenceClassification(PreTrainedModel):
    config_class = TinyTransformerConfig
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1  # Binarna klasifikácia – toxický / netoxický
        self.transformer = TinyTransformer(
            config.vocab_size,
            config.embed_dim,
            config.num_heads,
            config.ff_dim,
            config.num_layers
        )

    def forward(self, input_ids, attention_mask=None):
        # Forward pasáž cez transformer
        outputs = self.transformer(input_ids)
        return {"logits": outputs}

# Funkcia na načítanie modelu, tokenizéra a výber zariadenia
def load_tiny_model():
    device = torch.device("cpu")  # Použije CPU
    # Načíta konfiguračný súbor modelu z HuggingFace hubu
    config = TinyTransformerConfig.from_pretrained("AssistantsLab/Tiny-Toxic-Detector")
    # Načíta predtrénovaný model z repozitára
    model = TinyTransformerForSequenceClassification.from_pretrained(
        "AssistantsLab/Tiny-Toxic-Detector", config=config
    ).to(device)
    # Načíta predtrénovaný tokenizér
    tokenizer = AutoTokenizer.from_pretrained("AssistantsLab/Tiny-Toxic-Detector")
    return model, tokenizer, device

# Funkcia na predikciu toxicity a meranie latencie
def predict_tiny(text, model, tokenizer, device):
    start = time.perf_counter()  # Začiatok merania času

    # Tokenizácia vstupného textu s paddingom a truncovaním
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length").to(device)

    # Odstráni token_type_ids ak existuje (nepotrebné pri jedno-vstupových modeloch)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    # Deaktivuje výpočet gradientov počas inferencie
    with torch.no_grad():
        outputs = model(**inputs)

    # Výpočet latencie v milisekundách
    latency = (time.perf_counter() - start) * 1000
    # Výstup – pravdepodobnosť toxicity (sigmoid)
    logits = outputs["logits"].squeeze().item()
    return round(logits, 3), round(latency, 2)
