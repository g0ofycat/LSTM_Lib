# https://colab.research.google.com/

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter

import signal
import torch
import torch.nn as nn
import json
import re

# =========================
# // DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

torch.backends.cudnn.benchmark = True

# =========================
# // CONFIG
# =========================

VOCAB_SIZE = 6000
HIDDEN     = 1024
SEQ_LEN    = 128
BATCH_SIZE = 128
EPOCHS     = 30
LR         = 5e-4
GRAD_CLIP  = 1.0
DROPOUT    = 0.1
MIN_FREQ   = 3
EMBED_DIM  = 512
PAD        = "<pad>"
UNK        = "<unk>"

# =========================
# // SIGNAL HANDLER
# =========================

stop_training = False

def handle_sigterm(sig, frame):
    global stop_training
    print("\nCtrl+M - Saving...")
    stop_training = True

signal.signal(signal.SIGINT, handle_sigterm)

# =========================
# // VOCAB
# =========================

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ',!?.]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_vocab(texts):
    counts = Counter(tok for text in texts for tok in clean(text).split())
    top = sorted(counts, key=counts.get, reverse=True)
    top = [t for t in top if counts[t] >= MIN_FREQ][: VOCAB_SIZE - 2]
    itos = [PAD, UNK] + top
    stoi = {t: i for i, t in enumerate(itos)}
    return stoi, itos

def encode(text, stoi):
    unk_id = stoi[UNK]
    return [stoi.get(tok, unk_id) for tok in clean(text).split()]

# =========================
# // DATASET
# =========================

class TextDataset(Dataset):
    def __init__(self, texts, stoi):
        tokens = []
        for text in texts:
            tokens.extend(encode(text, stoi))

        n = (len(tokens) - 1) // SEQ_LEN * SEQ_LEN
        self.x = torch.tensor(tokens[:n]).view(-1, SEQ_LEN)
        self.y = torch.tensor(tokens[1:n+1]).view(-1, SEQ_LEN)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

# =========================
# // MODEL
# =========================

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, dropout):
        super().__init__()
        self.hidden = hidden
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.drop  = nn.Dropout(dropout)
        self.lstm  = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.Wy    = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        x = self.drop(self.embed(x))
        out, _ = self.lstm(x)
        return self.Wy(self.drop(out))

# =========================
# // TRAIN
# =========================

def run_epoch(model, loader, criterion, optimizer=None):
    model.train() if optimizer else model.eval()
    total_loss, total_tokens = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.set_grad_enabled(optimizer is not None):
            logits = model(x)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg = total_loss / total_tokens
    return avg, torch.exp(torch.tensor(avg)).item()

# =========================
# // EXPORT (JSON)
# =========================

def export(model, itos, path="lstm_export.json"):
    sd = model.state_dict()
    H  = model.hidden

    Wih = sd["lstm.weight_ih_l0"]
    Whh = sd["lstm.weight_hh_l0"]
    b   = sd["lstm.bias_ih_l0"] + sd["lstm.bias_hh_l0"]

    Wi, Wf, Wg, Wo = Wih.split(H, dim=0)
    Ui, Uf, Ug, Uo = Whh.split(H, dim=0)
    BI, BF, BG, BO = b.split(H, dim=0)

    def to_mat(t): return [[round(float(v), 4) for v in row] for row in t.cpu()]
    def to_col(t): return [[round(float(v), 4)] for v in t.cpu()]

    data = {
        "input_neurons":  len(itos),
        "hidden_neurons": H,
        "output_neurons": len(itos),
        "learning_rate":  LR,
        "dropout_rate":   0.0,
        "weights": [
            to_mat(Wf), to_mat(Uf),
            to_mat(Wi), to_mat(Ui),
            to_mat(Wg), to_mat(Ug),
            to_mat(Wo), to_mat(Uo),
            to_mat(sd["Wy.weight"])
        ],
        "bf": to_col(BF), "bi": to_col(BI),
        "bg": to_col(BG), "bo": to_col(BO),
        "by": to_col(sd["Wy.bias"]),
        "h":  to_col(torch.zeros(H)),
        "c":  to_col(torch.zeros(H)),
        "embed": to_mat(sd["embed.weight"]),
        "cache": {
            "inputs": [], "h": [], "c": [], "f": [], "i": [],
            "g": [],     "o": [], "y": [], "h_prev": [], "c_prev": []
        },
        "vocab": itos
    }

    with open(path, "w") as f:
        json.dump(data, f)

    print("Exported ->", path)

# =========================
# // LOAD DATA
# =========================

raw = load_dataset("ParlAI/blended_skill_talk")

def extract(split):
    texts = []
    for row in raw[split]:
        texts.extend(row["previous_utterance"])
        texts.extend(row["free_messages"])
        texts.extend(row["guided_messages"])
    return [t for t in texts if t.strip()]

train_texts = extract("train")
val_texts   = extract("validation")

stoi, itos = build_vocab(train_texts)

train_ds = TextDataset(train_texts, stoi)
val_ds   = TextDataset(val_texts, stoi)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# =========================
# // MODEL SETUP
# =========================

model = LSTMModel(len(itos), EMBED_DIM, HIDDEN, DROPOUT).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, min_lr=1e-5)

# =========================
# // TRAIN LOOP
# =========================

best_val = float("inf")

for epoch in range(1, EPOCHS + 1):
    if stop_training:
        break

    _, train_ppl = run_epoch(model, train_dl, criterion, optimizer)
    val_loss, val_ppl = run_epoch(model, val_dl, criterion)
    scheduler.step(val_loss)

    lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch} | train ppl {train_ppl:.1f} | val ppl {val_ppl:.1f} | lr {lr:.2e}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best.pt")

    if stop_training:
        print(f"Stopped after epoch {epoch}. Best checkpoint saved.")
        break

# =========================
# // EXPORT
# =========================

model.load_state_dict(torch.load("best.pt", weights_only=True))
export(model, itos)