"""
ToxScan AI — Sentence Transformer Classifier for In-Game Toxicity Detection
============================================================================
Joint intent classification on the CONDA dataset (CONtextual Dual-Annotated).

Intent classes (utterance-level):
    E = Explicit toxic
    I = Implicit toxic
    A = Action-based
    O = Other (non-toxic)

Architecture: Pretrained Sentence Transformer encoder (frozen) with a
2-layer MLP classification head over the pooled sentence embedding.

Why Sentence Transformer over custom BPE encoder:
  - Pretrained on hundreds of millions of sentence pairs; understands
    word order, negation, and context out of the box.
  - "I am happy, not sad" and "I am sad, not happy" get different embeddings.
  - Handles short/slang utterances (GG, tr4sh, FUK) without needing
    those exact strings in a training vocab.
  - Frozen encoder + small head = far fewer trainable parameters,
    which directly reduces overfitting on a small dataset like CONDA.

Class imbalance strategy:
  - WeightedRandomSampler: oversamples minority classes (I, A) each epoch.
  - Weighted CrossEntropyLoss: penalizes minority-class errors more.
  - Both together are stronger than either alone.

Math:
    Embedding:     e = SentenceTransformer(utterance)  ∈ R^d  (frozen)
    Head:          z = W2 · ReLU(W1 · e + b1) + b2
    Cross-entropy: L = -sum_c [ y_c · log(softmax(z)_c) ]

Usage:
    pip install sentence-transformers
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any


# ---------------------------------------------------------------------------
# Seeds & device
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_name": "toxscan_intent_classification",
        "task_type": "text_classification",
        "dataset": "CONDA (CONtextual Dual-Annotated)",
        "num_classes": 4,
        "label_map": {"E": 0, "I": 1, "A": 2, "O": 3},
        "model": "SentenceTransformerClassifier",
        "description": (
            "4-class utterance-level intent classification on in-game chat. "
            "Frozen pretrained Sentence Transformer + MLP head."
        ),
    }


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

LABEL_MAP   = {"E": 0, "I": 1, "A": 2, "O": 3}
LABEL_NAMES = ["Explicit", "Implicit", "Action", "Other"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CONDADataset(Dataset):
    """
    PyTorch Dataset wrapping a CONDA CSV split.

    Stores raw utterance strings and integer labels.
    No pre-encoding — the Sentence Transformer handles tokenization
    internally during the forward pass, so we just store text as-is.

    Expected columns: 'utterance', 'intentClass'
    """

    def __init__(self, texts: List[str], labels: List[int]):
        self.texts  = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], self.labels[idx]


def collate_fn(batch):
    """
    Custom collate: keeps texts as a plain list of strings (not a tensor)
    so the Sentence Transformer can receive them directly.
    """
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)


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
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df     = df.dropna(subset=["utterance", "intentClass"])
    df     = df[df["intentClass"].isin(LABEL_MAP)]
    texts  = df["utterance"].astype(str).tolist()
    labels = df["intentClass"].map(LABEL_MAP).tolist()
    return texts, labels


def make_dataloaders(
    data_dir:   str = "data",
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Load CONDA train/valid CSVs and return DataLoaders + class weights.

    Imbalance is handled two ways:
      1. WeightedRandomSampler  — each epoch sees a balanced class distribution.
      2. class_weights tensor   — passed to CrossEntropyLoss in main().

    Note: max_len and vocab are no longer needed; the Sentence Transformer
    handles its own tokenization and truncation internally.
    """
    train_path = os.path.join(data_dir, "CONDA_train.csv")
    valid_path = os.path.join(data_dir, "CONDA_valid.csv")

    for p in [train_path, valid_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected data file not found: {p}")

    print("Loading CONDA splits...")
    train_texts, train_labels = _load_split(train_path)
    val_texts,   val_labels   = _load_split(valid_path)

    print(f"  Train : {len(train_texts):,} samples")
    print(f"  Valid : {len(val_texts):,} samples")

    # Class distribution + weights
    from collections import Counter
    dist  = Counter(train_labels)
    inv   = {v: k for k, v in LABEL_MAP.items()}
    total = len(train_labels)
    print("  Train class distribution:")
    for idx in sorted(dist):
        print(f"    {inv[idx]} ({LABEL_NAMES[idx]}): {dist[idx]:,}")

    class_weights = torch.FloatTensor([
        total / (4 * dist.get(i, 1)) for i in range(4)
    ])

    # WeightedRandomSampler: each sample's weight = 1 / its class frequency.
    # This means minority classes (I, A) are sampled more often each epoch.
    sample_weights = [1.0 / dist[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )

    train_ds = CONDADataset(train_texts, train_labels)
    val_ds   = CONDADataset(val_texts,   val_labels)

    # sampler is mutually exclusive with shuffle=True
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, collate_fn=collate_fn,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0, pin_memory=False)

    return train_loader, val_loader, class_weights


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SentenceTransformerClassifier(nn.Module):
    """
    Frozen pretrained Sentence Transformer + trainable MLP classification head.

    The encoder is frozen: its weights never change during training.
    Only the two-layer MLP head is trained, which means:
      - Far fewer parameters to overfit (256*emb_dim + 256*4 vs full encoder).
      - Training is fast — embeddings can be computed without autograd.
      - Still benefits from rich pretrained representations.

    To fine-tune the full encoder instead, pass freeze_encoder=False.
    This requires more VRAM and a lower learning rate (~1e-5).

    Input  : list of raw utterance strings (batch)
    Output : (batch, 4) raw logits
    """

    # paraphrase-multilingual handles non-English player IDs and utterances
    # in the CONDA dataset (Vietnamese names, mixed-language chat, etc.)
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        model_name:     str   = DEFAULT_MODEL,
        num_classes:    int   = 4,
        freeze_encoder: bool  = True,
    ):
        super().__init__()
        print(f"  Loading pretrained encoder: {model_name}")
        self.encoder        = SentenceTransformer(model_name)
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("  Encoder frozen — training MLP head only.")

        # Infer embedding dimension from a single dummy forward pass
        with torch.no_grad():
            dummy = self.encoder.encode(["test"], convert_to_tensor=True)
        emb_dim = dummy.shape[-1]
        print(f"  Embedding dim: {emb_dim}")

        # Two-layer MLP head
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        texts   : list of raw utterance strings (one per sample in batch)
        returns : (batch, num_classes) logits
        """
        embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self.head(embeddings)


def build_model(device: torch.device) -> SentenceTransformerClassifier:
    """Instantiate and return the model on the correct device."""
    model = SentenceTransformerClassifier(
        model_name     = SentenceTransformerClassifier.DEFAULT_MODEL,
        num_classes    = 4,
        freeze_encoder = True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__} | Trainable params: {n_params:,}")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    criterion:    nn.Module,
    optimizer:    optim.Optimizer,
    device:       torch.device,
    epochs:       int  = 10,
    patience:     int  = 3,
    verbose:      bool = True,
) -> Dict[str, List[float]]:
    """Train with early stopping on validation macro-F1."""
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history      = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    best_f1      = 0.0
    best_state   = None
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for texts, labels in train_loader:          # texts is a list[str], not a tensor
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(texts)                   # model encodes internally
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
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

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

        if val_metrics["f1_macro"] > best_f1:
            best_f1      = val_metrics["f1_macro"]
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch} (best val F1={best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model:       nn.Module,
    data_loader: DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
) -> Dict[str, float]:
    """Evaluate model. Returns loss, accuracy, f1_macro, and per-class F1."""
    model.eval()
    total_loss             = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for texts, labels in data_loader:           # texts is list[str]
            labels = labels.to(device)
            logits = model(texts)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss     = total_loss / len(data_loader)
    accuracy     = accuracy_score(all_targets, all_preds)
    f1_macro     = f1_score(all_targets, all_preds, average="macro",   zero_division=0)
    f1_per_class = f1_score(all_targets, all_preds, average=None,
                            labels=[0, 1, 2, 3], zero_division=0)

    return {
        "loss":     avg_loss,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_E":     float(f1_per_class[0]),
        "f1_I":     float(f1_per_class[1]),
        "f1_A":     float(f1_per_class[2]),
        "f1_O":     float(f1_per_class[3]),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    model:      nn.Module,
    texts:      List[str],
    device:     torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Run inference on raw utterance strings.
    Returns integer class predictions (0=E, 1=I, 2=A, 3=O).
    """
    model.eval()
    all_preds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            preds = model(batch).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds)


def predict_proba(
    model:  nn.Module,
    texts:  List[str],
    device: torch.device,
) -> np.ndarray:
    """
    Return softmax probability distributions over the 4 intent classes.
    Shape: (N, 4)
    """
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(texts), dim=1)
    return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    model:      nn.Module,
    history:    Dict,
    metrics:    Dict,
    output_dir: str = "output",
    prefix:     str = "toxscan",
):
    """Save model head checkpoint, metrics JSON, and training curve."""
    os.makedirs(output_dir, exist_ok=True)

    # Save the full model state (frozen encoder buffers + trained head weights)
    ckpt_path = os.path.join(output_dir, f"{prefix}_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "label_map":        LABEL_MAP,
    }, ckpt_path)

    metrics_path = os.path.join(output_dir, f"{prefix}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    _save_training_curve(history, output_dir, prefix)

    print(f"\nArtifacts saved to '{output_dir}/'")


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
# Load saved model
# ---------------------------------------------------------------------------

def load_model(
    device:     torch.device,
    output_dir: str = "output",
    prefix:     str = "toxscan",
) -> nn.Module:
    """Load saved head checkpoint if it exists. Returns model or None."""
    ckpt_path = os.path.join(output_dir, f"{prefix}_model.pt")

    if not os.path.exists(ckpt_path):
        return None

    print(f"Found existing checkpoint — loading from '{output_dir}/'")
    model = SentenceTransformerClassifier().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Model loaded successfully — skipping training.")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ToxScan AI — CONDA intent classifier")
    parser.add_argument("--data_dir",   default="data",   help="Folder with train/valid CSV")
    parser.add_argument("--output_dir", default="output", help="Where to save artifacts")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=4)
    args = parser.parse_args()

    print("=" * 65)
    print("ToxScan AI — Sentence Transformer for In-Game Toxicity Detection")
    print("=" * 65)

    set_seed(42)
    device   = get_device()
    metadata = get_task_metadata()
    print(f"\nTask   : {metadata['task_name']}")
    print(f"Classes: {metadata['label_map']}")

    # ── Load or Train ──────────────────────────────────────────────────────
    model = load_model(device, output_dir=args.output_dir)

    if model is None:
        print("\n── Data loading ─────────────────────────────────────────────")
        train_loader, val_loader, class_weights = make_dataloaders(
            data_dir   = args.data_dir,
            batch_size = args.batch_size,
        )

        print("\n── Model ────────────────────────────────────────────────────")
        model     = build_model(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

        print("\n── Training ─────────────────────────────────────────────────")
        history = train(
            model, train_loader, val_loader,
            criterion, optimizer, device,
            epochs  = args.epochs,
            patience= args.patience,
        )

        print("\n── Evaluation ───────────────────────────────────────────────")
        train_metrics = evaluate(model, train_loader, criterion, device)
        val_metrics   = evaluate(model, val_loader,   criterion, device)

        print(f"\n{'Split':<10} {'Loss':>8} {'Acc':>8} {'F1-macro':>10}")
        print("-" * 40)
        for split, m in [("Train", train_metrics), ("Val", val_metrics)]:
            print(f"{split:<10} {m['loss']:>8.4f} {m['accuracy']:>8.4f} {m['f1_macro']:>10.4f}")
        print(f"\nPer-class F1 (val):")
        for cls, key in zip(LABEL_NAMES, ["f1_E", "f1_I", "f1_A", "f1_O"]):
            print(f"  {cls:<12}: {val_metrics[key]:.4f}")

        save_artifacts(model, history,
                       {"train": train_metrics, "validation": val_metrics,
                        "metadata": metadata},
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
    preds = predict(model, sample_utterances, device)
    probs = predict_proba(model, sample_utterances, device)
    print(f"\n{'Utterance':<42} {'Pred':>6}  {'Conf':>6}")
    print("-" * 58)
    for utt, pred, prob in zip(sample_utterances, preds, probs):
        label = inv_label[int(pred)]
        conf  = prob[int(pred)]
        print(f"{utt[:40]:<42} {label:>6}  {conf:>6.2%}")

    # ── Quality checks ─────────────────────────────────────────────────────
    if "history" in locals():
        print("\n" + "=" * 65)
        print("QUALITY CHECKS")
        print("=" * 65)
        checks = [
            ("Val   accuracy  > 0.60", val_metrics["accuracy"]  > 0.60, val_metrics["accuracy"]),
            ("Val   F1-macro  > 0.50", val_metrics["f1_macro"]  > 0.50, val_metrics["f1_macro"]),
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
        print("PASS" if all_passed else "FAIL — one or more quality checks failed.")
        print("=" * 65)

    # ── Interactive loop ───────────────────────────────────────────────────
    while True:
        print()
        user_input = input("Enter a string to evaluate or q to quit: ")
        if user_input.lower() == "q":
            break

        preds_i = predict(model, [user_input], device)
        probs_i = predict_proba(model, [user_input], device)
        label   = inv_label[int(preds_i[0])]
        conf    = probs_i[0][int(preds_i[0])]
        print(f"\n{'Utterance':<42} {'Pred':>6}  {'Conf':>6}")
        print("-" * 58)
        print(f"{user_input[:40]:<42} {label:>6}  {conf:>6.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())