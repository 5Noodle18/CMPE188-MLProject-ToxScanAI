"""
ToxScan AI — Transformer Encoder Classifier for In-Game Toxicity Detection
============================================================================
Joint intent classification on the CONDA dataset (CONtextual Dual-Annotated).

Intent classes (utterance-level):
    E = Explicit toxic
    I = Implicit toxic
    A = Action-based
    O = Other (non-toxic)

Architecture: Transformer encoder (tfm_lvl2 scaffold) with a 4-way softmax
classification head over the [CLS] token representation.

Math:
    Attention:      Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    Cross-entropy:  L = -sum_c [ y_c * log(softmax(z)_c) ]

Usage:
    python src/CONDA.py                       # expects data/ in project root
    python src/CONDA.py --data_dir path/to/data
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# ---------------------------------------------------------------------------
# Seeds & device
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42): #42 is the answer to life, the universe, and everything
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:   # try to retrieve a GPU if available, else fall back to CPU
    """Return CUDA if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def get_task_metadata() -> Dict[str, Any]: #if we are using a json file to store metadata
    """Return task metadata."""
    return {
        "task_name": "toxscan_intent_classification",
        "task_type": "text_classification",
        "dataset": "CONDA (CONtextual Dual-Annotated)",
        "num_classes": 4,
        "label_map": {"E": 0, "I": 1, "A": 2, "O": 3},
        "model": "TransformerEncoderClassifier",
        "description": (
            "4-class utterance-level intent classification on in-game chat. "
            "Transformer encoder over a learned vocabulary with [CLS] pooling."
        ),
    }


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """Simple word-level vocabulary with <PAD> and <UNK> tokens."""

    PAD_IDX = 0  # Padding token for sequence batching (fixed index for embedding layer)
    UNK_IDX = 1  # Unkown token for out-of-vocab words

    def __init__(self, max_size: int = 15000):
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1} # word to index mapping, starting with special tokens
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"} # index to word mapping, starting with special tokens
        self.max_size = max_size # maximum vocabulary size (including special tokens)

    # this creates a vocabulary from a list of raw text strings, counting word frequencies and keeping the most common ones up to max_size (accounting for special tokens)
    def build(self, texts: List[str], min_freq: int = 1):
        from collections import Counter
        counts = Counter(w for text in texts for w in text.lower().split()) # count word frequencies across all texts (lowercased, split on whitespace)
        for word, freq in counts.most_common(self.max_size - 2): # iterate over most common words up to max_size (accounting for 2 special tokens)
            if freq >= min_freq: 
                idx = len(self.word2idx)    # assign next available index
                self.word2idx[word] = idx   # add word to word2idx mapping
                self.idx2word[idx] = word   # add index to idx2word mapping

    # this converts a raw text string into a list of token ids, using the vocabulary mapping and padding to max_len
    def encode(self, text: str, max_len: int) -> List[int]:
        tokens = text.lower().split()[:max_len]
        ids = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        ids += [self.PAD_IDX] * (max_len - len(ids))   # pad to max_len
        return ids

    # this returns the size of the vocabulary (number of unique tokens including special tokens)
    def __len__(self) -> int:
        return len(self.word2idx) 


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

LABEL_MAP = {"E": 0, "I": 1, "A": 2, "O": 3}    # From intent class labels to integer indices for model training and evaluation
LABEL_NAMES = ["Explicit", "Implicit", "Action", "Other"]


class CONDADataset(Dataset):
    """
    PyTorch Dataset wrapping a CONDA CSV split.

    Expected columns: 'utterance', 'intentClass' Id,matchId,conversationId,utterance,chatTime,playerSlot,playerId

    """

    def __init__(self, texts: List[str], labels: List[int],
                 vocab: Vocabulary, max_len: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    # this retrieves the token ids and label for a single data point at index idx, encoding the text using the vocabulary and returning tensors for model input
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.vocab.encode(self.texts[idx], self.max_len)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_split(path: str) -> Tuple[List[str], List[int]]:
    """Load a single CSV split; return (texts, labels)."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")

    df = pd.read_csv(path)

    required = {"utterance", "intentClass"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df = df.dropna(subset=["utterance", "intentClass"])
    df = df[df["intentClass"].isin(LABEL_MAP)]          # drop any unlabelled rows

    texts  = df["utterance"].astype(str).tolist()
    labels = df["intentClass"].map(LABEL_MAP).tolist()
    return texts, labels


def make_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    max_len: int = 64,
    min_vocab_freq: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    """
    Load CONDA train/valid/test CSVs and return DataLoaders + Vocabulary.

    Expects:
        <data_dir>/train.csv
        <data_dir>/valid.csv
        <data_dir>/test.csv
    """
    train_path = os.path.join(data_dir, "CONDA_train.csv")
    valid_path = os.path.join(data_dir, "CONDA_valid.csv")
    test_path  = os.path.join(data_dir, "CONDA_test.csv")

    for p in [train_path, valid_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected data file not found: {p}")

    print("Loading CONDA splits...")
    train_texts, train_labels = _load_split(train_path)
    val_texts,   val_labels   = _load_split(valid_path)
   # test_texts,  test_labels  = _load_split(test_path) #ommitted because test is unnanotated 

    print(f"  Train : {len(train_texts):,} samples")
    print(f"  Valid : {len(val_texts):,} samples")
   # print(f"  Test  : {len(test_texts):,} samples")

    # Build vocabulary on training data only (no leakage)
    vocab = Vocabulary(max_size=15000)
    vocab.build(train_texts, min_freq=min_vocab_freq)
    print(f"  Vocabulary size: {len(vocab):,} tokens")

    # Print class distribution
    from collections import Counter
    dist = Counter(train_labels)
    inv = {v: k for k, v in LABEL_MAP.items()}
    print("  Train class distribution:")
    for idx in sorted(dist):
        print(f"    {inv[idx]} ({LABEL_NAMES[idx]}): {dist[idx]:,}")

    # Compute class weights for imbalanced loss
    total = len(train_labels)
    weights = torch.FloatTensor([
        total / (4 * dist.get(i, 1)) for i in range(4)
    ])

    def make_loader(texts, labels, shuffle):
        ds = CONDADataset(texts, labels, vocab, max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    train_loader = make_loader(train_texts, train_labels, shuffle=True)
    val_loader   = make_loader(val_texts,   val_labels,   shuffle=False)
   # test_loader  = make_loader(test_texts,  test_labels,  shuffle=False)

    return train_loader, val_loader, vocab, weights #normallly, include test_loader as 3rd arg


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _EncoderLayer(nn.Module):
    """Single pre-norm transformer encoder layer."""

    def __init__(self, d_model: int, num_heads: int,
                 dim_ff: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, num_heads,
                                           dropout=dropout, batch_first=False)
        self.ff1   = nn.Linear(d_model, dim_ff)
        self.ff2   = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm self-attention
        x2 = self.norm1(x)
        x2, _ = self.attn(x2, x2, x2, key_padding_mask=key_padding_mask)
        x = x + self.drop(x2)
        # Pre-norm feed-forward
        x2 = self.norm2(x)
        x2 = self.ff2(self.drop(torch.relu(self.ff1(x2))))
        x = x + self.drop(x2)
        return x


class TransformerEncoderClassifier(nn.Module):
    """
    Encoder-only transformer for 4-class intent classification.

    Input  : (batch, seq_len)  — token ids
    Output : (batch, 4)        — raw logits (pass through CrossEntropyLoss)
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 num_heads: int = 4, num_layers: int = 2,
                 dim_ff: int = 256, num_classes: int = 4,
                 max_len: int = 64, dropout: float = 0.1,
                 pad_idx: int = 0):
        super().__init__()
        self.pad_idx   = pad_idx
        self.d_model   = d_model
        self.max_len   = max_len
        self.vocab_size = vocab_size

        self.embedding   = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc     = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        self.drop        = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            _EncoderLayer(d_model, num_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm       = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.classifier.weight, -0.1, 0.1)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        key_padding_mask = (input_ids == self.pad_idx)       # (batch, seq_len)

        x = self.embedding(input_ids)                        # (batch, seq_len, d_model)
        x = x + self.pos_enc[:, :x.size(1), :]
        x = self.drop(x)

        x = x.transpose(0, 1)                               # (seq_len, batch, d_model)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        cls = x[0]                                           # [CLS] = first token position
        return self.classifier(self.drop(cls))               # (batch, num_classes)


def build_model(vocab: Vocabulary, device: torch.device,
                max_len: int = 64) -> TransformerEncoderClassifier:
    """Instantiate and return the model on the correct device."""
    model = TransformerEncoderClassifier(
        vocab_size  = len(vocab),
        d_model     = 128,
        num_heads   = 4,
        num_layers  = 2,
        dim_ff      = 256,
        num_classes = 4,
        max_len     = max_len,
        dropout     = 0.1,
        pad_idx     = Vocabulary.PAD_IDX,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__} | "
          f"Params: {n_params:,} | Vocab: {len(vocab):,}")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
    patience: int = 3,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Train with early stopping on validation macro-F1.
    Returns training history dict.
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    best_f1      = 0.0
    best_state   = None
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total

        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["f1_macro"])
        print(f"Current LR: {optimizer.param_groups[0]['lr']}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1_macro"])
        history["val_acc"].append(val_metrics["accuracy"])

        if verbose:
            print(f"Epoch {epoch:>2}/{epochs} | "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f}  "
                  f"val_acc={val_metrics['accuracy']:.4f}  "
                  f"val_f1={val_metrics['f1_macro']:.4f}")

        # Early stopping
        if val_metrics["f1_macro"] > best_f1:
            best_f1    = val_metrics["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch} (best val F1={best_f1:.4f})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model. Returns dict with loss, accuracy, f1_macro,
    and per-class f1 scores.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            logits = model(input_ids)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    f1_per_class = f1_score(all_targets, all_preds, average=None,
                            labels=[0, 1, 2, 3], zero_division=0)

    return {
        "loss":      avg_loss,
        "accuracy":  accuracy,
        "f1_macro":  f1_macro,
        "f1_E":      float(f1_per_class[0]),
        "f1_I":      float(f1_per_class[1]),
        "f1_A":      float(f1_per_class[2]),
        "f1_O":      float(f1_per_class[3]),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    model: nn.Module,
    texts: List[str],
    vocab: Vocabulary,
    device: torch.device,
    max_len: int = 64,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Run inference on a list of raw utterance strings.
    Returns integer class predictions (0=E, 1=I, 2=A, 3=O).
    """
    model.eval()
    all_preds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        ids   = [vocab.encode(t, max_len) for t in batch]
        t     = torch.tensor(ids, dtype=torch.long).to(device)
        with torch.no_grad():
            preds = model(t).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds)


def predict_proba(
    model: nn.Module,
    texts: List[str],
    vocab: Vocabulary,
    device: torch.device,
    max_len: int = 64,
) -> np.ndarray:
    """
    Return softmax probability distributions over the 4 intent classes.
    Shape: (N, 4)
    """
    model.eval()
    ids = [vocab.encode(t, max_len) for t in texts]
    t   = torch.tensor(ids, dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(t)
        probs  = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    model: nn.Module,
    vocab: Vocabulary,
    history: Dict,
    metrics: Dict,
    output_dir: str = "output",
    prefix: str = "toxscan",
):
    """Save model checkpoint, vocabulary, metrics JSON, and training curve."""
    os.makedirs(output_dir, exist_ok=True)

    # Model checkpoint
    ckpt_path = os.path.join(output_dir, f"{prefix}_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size":       model.vocab_size,
        "d_model":          model.d_model,
        "max_len":          model.max_len,
        "num_classes":      4,
        "label_map":        LABEL_MAP,
    }, ckpt_path)

    # Vocabulary
    vocab_path = os.path.join(output_dir, f"{prefix}_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"word2idx": vocab.word2idx}, f)

    # Metrics
    metrics_path = os.path.join(output_dir, f"{prefix}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Training curve
    _save_training_curve(history, output_dir, prefix)

    print(f"\nArtifacts saved to '{output_dir}/'")
    print(f"  {os.path.basename(ckpt_path)}")
    print(f"  {os.path.basename(vocab_path)}")
    print(f"  {os.path.basename(metrics_path)}")
    print(f"  {prefix}_training_curve.png")


def _save_training_curve(history: Dict, output_dir: str, prefix: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train loss")
    ax1.plot(history["val_loss"],   label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss curves"); ax1.legend()

    ax2.plot(history["val_acc"], label="Val accuracy")
    ax2.plot(history["val_f1"],  label="Val macro-F1")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
    ax2.set_title("Validation metrics"); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_training_curve.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def _save_classification_report(all_targets, all_preds, output_dir, prefix):
    report = classification_report(
        all_targets, all_preds,
        target_names=LABEL_NAMES,
        zero_division=0,
    )
    report_path = os.path.join(output_dir, f"{prefix}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print("\nClassification report:")
    print(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ToxScan AI — CONDA intent classifier")
    parser.add_argument("--data_dir",   default="data",   help="Path to folder with train/valid/test CSV")
    parser.add_argument("--output_dir", default="output", help="Where to save artifacts")
    parser.add_argument("--epochs",     type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--max_len",    type=int, default=64)
    parser.add_argument("--patience",   type=int, default=4)
    args = parser.parse_args()

    print("=" * 65)
    print("ToxScan AI — Transformer Encoder for In-Game Toxicity Detection")
    print("=" * 65)

    set_seed(42)
    device   = get_device()
    metadata = get_task_metadata()
    print(f"\nTask   : {metadata['task_name']}")
    print(f"Classes: {metadata['label_map']}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n── Data loading ─────────────────────────────────────────────")
    train_loader, val_loader, vocab, class_weights = make_dataloaders( #test_loader as 3rd argument if we were using annotated test
        data_dir   = args.data_dir,
        batch_size = args.batch_size,
        max_len    = args.max_len,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n── Model ────────────────────────────────────────────────────")
    model     = build_model(vocab, device, max_len=args.max_len)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # ── Training ──────────────────────────────────────────────────────────
    print("\n── Training ─────────────────────────────────────────────────")
    history = train(
        model, train_loader, val_loader,
        criterion, optimizer, device,
        epochs  = args.epochs,
        patience= args.patience,
    )

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n── Evaluation ───────────────────────────────────────────────")
    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics   = evaluate(model, val_loader,   criterion, device)
   # test_metrics  = evaluate(model, test_loader,  criterion, device)

    print(f"\n{'Split':<10} {'Loss':>8} {'Acc':>8} {'F1-macro':>10}")
    print("-" * 40)
    for split, m in [("Train", train_metrics), ("Val", val_metrics)]: #, ("Test", test_metrics)]:
        print(f"{split:<10} {m['loss']:>8.4f} {m['accuracy']:>8.4f} {m['f1_macro']:>10.4f}")

 #   print(f"\nPer-class F1 on test set:")
 #   for cls, key in zip(LABEL_NAMES, ["f1_E", "f1_I", "f1_A", "f1_O"]):
 #       print(f"  {cls:<10}: {test_metrics[key]:.4f}")

    # Full classification report on test set
    model.eval()
    all_preds, all_targets = [], []
 #   with torch.no_grad():
 #       for input_ids, labels in test_loader:
 #           logits = model(input_ids.to(device))
 #           all_preds.extend(logits.argmax(1).cpu().numpy())
 #           all_targets.extend(labels.numpy())
 #   _save_classification_report(all_targets, all_preds, args.output_dir, "toxscan")

    # ── Artifacts ─────────────────────────────────────────────────────────
    all_metrics = {
        "train": train_metrics,
        "validation": val_metrics,
 #       "test": test_metrics,
        "metadata": metadata,
    }
    save_artifacts(model, vocab, history, all_metrics,
                   output_dir=args.output_dir, prefix="toxscan")

    # ── Quick inference demo ───────────────────────────────────────────────
    print("\n── Inference demo ───────────────────────────────────────────")
    inv_label = {v: k for k, v in LABEL_MAP.items()}
    sample_utterances = [
        "gg wp well played everyone",
        "you are so trash get out of this game",
        "reported you for feeding",
        "push mid now",
    ]
    preds = predict(model, sample_utterances, vocab, device, max_len=args.max_len)
    probs = predict_proba(model, sample_utterances, vocab, device, max_len=args.max_len)
    print(f"\n{'Utterance':<42} {'Pred':>6}  {'Conf':>6}")
    print("-" * 58)
    for utt, pred, prob in zip(sample_utterances, preds, probs):
        label = inv_label[int(pred)]
        conf  = prob[int(pred)]
        print(f"{utt[:40]:<42} {label:>6}  {conf:>6.2%}")

    # ── Quality checks ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("QUALITY CHECKS")
    print("=" * 65)

    checks = [
        ("Val   accuracy  > 0.60", val_metrics["accuracy"]  > 0.60, val_metrics["accuracy"]),
        ("Val   F1-macro  > 0.50", val_metrics["f1_macro"]  > 0.50, val_metrics["f1_macro"]),
 #       ("Test  accuracy  > 0.55", test_metrics["accuracy"] > 0.55, test_metrics["accuracy"]),
 #       ("Test  F1-macro  > 0.45", test_metrics["f1_macro"] > 0.45, test_metrics["f1_macro"]),
        ("Train loss decreased",
            history["train_loss"][-1] < history["train_loss"][0],
            f"{history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f}"),
        ("No severe overfit (val_acc gap < 0.20)",
            abs(train_metrics["accuracy"] - val_metrics["accuracy"]) < 0.20,
            abs(train_metrics["accuracy"] - val_metrics["accuracy"])),
    ]

    all_passed = True
    for desc, passed, value in checks:
        sym = "!OOO!" if passed else "!XXX!"
        print(f"  {sym} {desc}: {value}")
        all_passed = all_passed and passed

    print("\n" + "=" * 65)
    if all_passed:
        print("PASS — all quality checks passed!")
    else:
        print("FAIL — one or more quality checks failed.")
    print("=" * 65)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())