"""
ToxScan AI — Transformer Encoder Classifier for In-Game Toxicity Detection
============================================================================
Joint intent classification on the CONDA dataset (CONtextual Dual-Annotated).

Intent classes (utterance-level):
    E = Explicit toxic
    I = Implicit toxic
    A = Action-based
    O = Other (non-toxic)

Architecture: Transformer encoder with:
  - BPE subword tokenization  (handles OOV, typos, slang)
  - spaCy lemmatization       (morphological normalization)
  - Character n-gram embeddings (char-level typo robustness)
  - TF-IDF soft attention bias (corpus-level word importance signal)
  - 4-way softmax classification head over [CLS] token

Math:
    Attention:      Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + B_tfidf) * V
    Cross-entropy:  L = -sum_c [ y_c * log(softmax(z)_c) ]
    Char n-gram emb: e(w) = (1/|N(w)|) * sum_{g in N(w)} E_char[g]   (mean pooling)

Usage:
    pip install tokenizers spacy scikit-learn
    python -m spacy download en_core_web_sm
    python src/CONDA.py                       # expects data/ in project root
    python src/CONDA.py --data_dir path/to/data
"""

import tempfile
import os
import sys
import json
import argparse
import re
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

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
        "model": "TransformerEncoderClassifier",
        "description": (
            "4-class utterance-level intent classification on in-game chat. "
            "BPE subword tokens + char n-gram embeddings + TF-IDF bias."
        ),
    }


LABEL_MAP   = {"E": 0, "I": 1, "A": 2, "O": 3}
LABEL_NAMES = ["Explicit", "Implicit", "Action", "Other"]

# ---------------------------------------------------------------------------
# Lemmatizer (spaCy)
# ---------------------------------------------------------------------------

class Lemmatizer:
    """
    Wraps spaCy for lemmatization.  Falls back to lowercasing if spaCy is
    unavailable (no crash, just a warning).

    Purpose:
      - 'reported', 'reporting', 'reports' → 'report'
      - Shrinks effective vocab; helps the BPE tokenizer see shared subwords.
    """

    def __init__(self):
        self._nlp = None
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm",
                                   disable=["parser", "ner", "senter"])
            print("  Lemmatizer: spaCy en_core_web_sm loaded.")
        except Exception as e:
            print(f"  Lemmatizer: spaCy unavailable ({e}). Using lower() fallback.")

    def lemmatize(self, text: str) -> str:
        """Return lemmatized version of text, preserving whitespace-split tokens."""
        if self._nlp is None:
            return text.lower()
        doc = self._nlp(text.lower())
        return " ".join(tok.lemma_ for tok in doc)

    def batch_lemmatize(self, texts: List[str],
                        batch_size: int = 512) -> List[str]:
        """Efficient batch lemmatization using spaCy's nlp.pipe."""
        if self._nlp is None:
            return [t.lower() for t in texts]
        results = []
        for doc in self._nlp.pipe(
            (t.lower() for t in texts), batch_size=batch_size
        ):
            results.append(" ".join(tok.lemma_ for tok in doc))
        return results


# ---------------------------------------------------------------------------
# BPE Subword Tokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer trained on the training corpus.

    Purpose:
      - Splits rare/OOV words into known subword pieces.
        'tr4sh' → ['tr', '##4', '##sh']  — avoids previous <UNK> black-hole.
      - Naturally handles typos, slang, and gaming neologisms.
      - vocab_size is tunable: larger = finer pieces, smaller = more sharing.

    Falls back to character-level if the `tokenizers` library is missing.
    """

    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size            # target vocab size for BPE training
        self._tokenizer = None                  # HuggingFace tokenizers object (if available)
        # Fallback: simple char-level
        self._char2idx: Dict[str, int] = {}     # character level fallback
        self._use_hf = False                    # bool flag to know if we're using HuggingFace tokenizer or char-level fallback

    # ── Training ────────────────────────────────────────────────────────────

    def train(self, texts: List[str], tmp_path: str = None): # normal training
        """Train BPE on a list of strings."""
        try:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import Whitespace

            if tmp_path is None: #cross-platform support for temporary file creation
                fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="_bpe_corpus_")
                os.close(fd)  # close file descriptor, we just need the path

            # Write corpus to temp file (tokenizers lib expects files)
            with open(tmp_path, "w", encoding="utf-8") as f:
                for t in texts:
                    f.write(t + "\n")

            tokenizer = Tokenizer(BPE(unk_token=self.UNK_TOKEN))
            tokenizer.pre_tokenizer = Whitespace()

            trainer = BpeTrainer(
                vocab_size      = self.vocab_size,
                special_tokens  = [self.PAD_TOKEN, self.UNK_TOKEN, self.CLS_TOKEN],
                min_frequency   = 2,
                show_progress   = False,
            )
            tokenizer.train([tmp_path], trainer)

            # Add [CLS] prepending as a post-processor
            cls_id = tokenizer.token_to_id(self.CLS_TOKEN)
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{self.CLS_TOKEN} $A",
                special_tokens=[(self.CLS_TOKEN, cls_id)],
            )

            self._tokenizer = tokenizer
            self._use_hf    = True
            print(f"  BPE tokenizer trained | vocab size: {tokenizer.get_vocab_size():,}")
        except ImportError:
            print("  'tokenizers' lib not found — using char-level fallback.")
            self._train_char_fallback(texts)

    def _train_char_fallback(self, texts: List[str]):   # fallback training
        """Character-level fallback when HuggingFace tokenizers is absent."""
        chars = set()
        for t in texts:
            chars.update(t)
        specials = [self.PAD_TOKEN, self.UNK_TOKEN, self.CLS_TOKEN]
        vocab = specials + sorted(chars)
        self._char2idx = {c: i for i, c in enumerate(vocab)}
        self._use_hf = False
        print(f"  Char-level fallback | vocab size: {len(self._char2idx):,}")

    # ── Encode ──────────────────────────────────────────────────────────────

    def encode(self, text: str, max_len: int) -> List[int]:
        """
        Encode a single string → padded list of token ids (length = max_len).
        [CLS] is always prepended.
        """
        if self._use_hf:
            enc = self._tokenizer.encode(text)
            ids = enc.ids[:max_len]
        else:
            ids = [self._char2idx.get(self.CLS_TOKEN, 2)]
            for ch in text[:max_len - 1]:
                ids.append(self._char2idx.get(ch, 1))  # 1 = UNK

        pad_id = self.pad_id
        ids += [pad_id] * (max_len - len(ids))
        return ids

    def batch_encode(self, texts: List[str], max_len: int) -> List[List[int]]:
        if self._use_hf:
            self._tokenizer.enable_padding(pad_id=self.pad_id,
                                           pad_token=self.PAD_TOKEN,
                                           length=max_len)
            self._tokenizer.enable_truncation(max_length=max_len)
            return [enc.ids for enc in self._tokenizer.encode_batch(texts)]
        else:
            return [self.encode(t, max_len) for t in texts]

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str):  #save the tokenizer to a file
        if self._use_hf:
            self._tokenizer.save(path)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"char2idx": self._char2idx, "use_hf": False}, f)

    def load(self, path: str):  #load the tokenizer from a file 
        if path.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "char2idx" in raw:
                self._char2idx = raw["char2idx"]
                self._use_hf   = False
                return
        try:
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(path)
            self._use_hf    = True
        except Exception:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._char2idx = data["char2idx"]
            self._use_hf   = False

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def vocab_size_actual(self) -> int:
        if self._use_hf:
            return self._tokenizer.get_vocab_size()
        return len(self._char2idx)

    @property
    def pad_id(self) -> int:
        if self._use_hf:
            return self._tokenizer.token_to_id(self.PAD_TOKEN)
        return self._char2idx.get(self.PAD_TOKEN, 0)

    @property
    def unk_id(self) -> int:
        if self._use_hf:
            return self._tokenizer.token_to_id(self.UNK_TOKEN)
        return self._char2idx.get(self.UNK_TOKEN, 1)


# ---------------------------------------------------------------------------
# Character N-gram Embedding
# ---------------------------------------------------------------------------

class CharNgramEmbedding(nn.Module):
    """
    Character n-gram embedding layer.

    Purpose:
      Words that share character sequences share embedding signal.
      'trash', 'tr4sh', 'traash' all activate overlapping n-grams.
      This gives the model robustness to typos and leet-speak without
      needing those exact strings in the training vocab.

    Implementation:
      For each subword token (from BPE), we hash its character n-grams into
      a fixed-size bucket table (à la FastText).  The token's char embedding
      is the mean of its n-gram bucket embeddings.

      e(token) = mean_pool( E_char[hash(g) % B] for g in ngrams(token, n_min..n_max) )

    This table is separate from the main token embedding and the two are summed:
      final_emb(token) = E_token[id] + e(token_string)
    """

    def __init__(self, num_buckets: int = 20000, d_model: int = 128,
                 n_min: int = 2, n_max: int = 4): #define an n-gram of min 2 and max 4 characters, and a bucket size of 20k
        super().__init__()
        self.num_buckets = num_buckets
        self.n_min       = n_min
        self.n_max       = n_max
        self.bucket_emb  = nn.Embedding(num_buckets, d_model)
        nn.init.uniform_(self.bucket_emb.weight, -0.05, 0.05)

    def _ngrams(self, token: str) -> List[str]:
        """Extract character n-grams from a token string."""
        token = f"<{token}>"   # boundary markers
        return [
            token[i:i+n]
            for n in range(self.n_min, self.n_max + 1)
            for i in range(len(token) - n + 1)
        ]

    def _hash_ngram(self, gram: str) -> int:
        """Deterministic bucket hash for a character n-gram."""
        return int(hashlib.md5(gram.encode()).hexdigest(), 16) % self.num_buckets

    def precompute_token_ngram_ids(
        self, token_strings: List[str]
    ) -> torch.Tensor:
        """
        Precompute a (vocab_size, max_ngrams) tensor of bucket ids.
        Padded with 0 where a token has fewer n-grams than max_ngrams.
        Called once after tokenizer training; stored as a buffer.
        """
        all_ids = []
        for tok in token_strings:
            grams  = self._ngrams(tok)
            ids    = [self._hash_ngram(g) for g in grams] if grams else [0]
            all_ids.append(ids)

        max_ng = max(len(x) for x in all_ids)
        padded = np.zeros((len(all_ids), max_ng), dtype=np.int64)
        for i, ids in enumerate(all_ids):
            padded[i, :len(ids)] = ids
        return torch.tensor(padded, dtype=torch.long)   # (V, max_ng)

    def forward(
        self,
        token_ids: torch.Tensor,              # (batch, seq)
        ngram_ids: torch.Tensor,              # (vocab, max_ng)  — precomputed buffer
    ) -> torch.Tensor:
        """
        Returns char n-gram embedding for each token in the batch.
        Shape: (batch, seq, d_model)
        """
        # Gather n-gram bucket ids for each token
        # token_ids: (batch, seq)  →  ngram_ids[token_ids]: (batch, seq, max_ng)
        tok_ng   = ngram_ids[token_ids]                        # (B, S, max_ng)
        emb_ng   = self.bucket_emb(tok_ng)                     # (B, S, max_ng, d)
        # Mean-pool across n-grams (ignore padding zeros by treating them as zero-vecs)
        mask     = (tok_ng != 0).unsqueeze(-1).float()         # (B, S, max_ng, 1)
        char_emb = (emb_ng * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        return char_emb                                        # (B, S, d)


# ---------------------------------------------------------------------------
# TF-IDF Attention Bias
# ---------------------------------------------------------------------------

class TFIDFAttentionBias:
    """
    Computes a per-token TF-IDF score from the training corpus, then injects
    it as an additive bias into transformer attention.

    Purpose:
      High-IDF words (rare but discriminative — slurs, specific gaming terms)
      get boosted attention weight.  Common words like 'the', 'a' are suppressed.
      Avoids unnecessary attention to common words and gives a strong prior toward important words.

    How it's used:
      The bias is a scalar per token, broadcast into attention logits:
        Attention(Q,K,V) = softmax( QK^T/sqrt(d) + tfidf_bias ) * V

      This is a *soft* routing signal — the attention heads can still override it,
      but they have a prior toward important words.
    """

    def __init__(self, max_features: int = 10000):
        self.max_features = max_features
        self._vectorizer  = TfidfVectorizer(
            max_features = max_features,
            sublinear_tf = True,       # log(1 + tf) dampens high-freq terms
            analyzer     = "word",
            token_pattern= r"(?u)\b\w+\b",
        )
        self._word2idf: Dict[str, float] = {}

    def fit(self, texts: List[str]):
        """Fit TF-IDF on training texts and store word→IDF mapping."""
        self._vectorizer.fit(texts)
        vocab = self._vectorizer.vocabulary_        # word → column index
        idf   = self._vectorizer.idf_              # IDF array (one per feature)
        self._word2idf = {w: float(idf[i]) for w, i in vocab.items()}
        print(f"  TF-IDF bias: {len(self._word2idf):,} terms fitted.")

    def score(self, token_string: str) -> float:
        """IDF score for a single token string (0 if OOV)."""
        return self._word2idf.get(token_string.lower(), 0.0)

    def score_sequence(self, token_strings: List[str]) -> List[float]:
        return [self.score(t) for t in token_strings]

    def precompute_vocab_scores(
        self, token_strings: List[str]
    ) -> torch.Tensor:
        """
        Precompute IDF score for every vocabulary token.
        Returns tensor of shape (vocab_size,) for use as an embedding offset.
        """
        scores = [self.score(t) for t in token_strings]
        return torch.tensor(scores, dtype=torch.float32)  # (V,)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CONDADataset(Dataset):
    """
    PyTorch Dataset wrapping a CONDA CSV split.

    Stores pre-encoded token id tensors and pre-computed TF-IDF bias vectors
    so that per-batch collation is fast.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BPETokenizer,
        max_len: int,
        tfidf_vocab_scores: Optional[torch.Tensor] = None,
    ):
        self.labels             = labels
        self.max_len            = max_len
        self.tfidf_vocab_scores = tfidf_vocab_scores   # (V,)  — may be None

        # Pre-encode all texts up front (batch encode is fast with HF tokenizers)
        encoded = tokenizer.batch_encode(texts, max_len)
        self.input_ids = [torch.tensor(ids, dtype=torch.long) for ids in encoded]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], torch.tensor(self.labels[idx], dtype=torch.long)


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
    data_dir: str      = "data",
    batch_size: int    = 32,
    max_len: int       = 64,
    bpe_vocab_size: int = 8000,
) -> Tuple[DataLoader, DataLoader, BPETokenizer, TFIDFAttentionBias, torch.Tensor, torch.Tensor]:
    """
    Load CONDA train/valid CSVs, preprocess, and return DataLoaders.

    Pipeline:
      1. Load raw texts
      2. Lemmatize with spaCy
      3. Train BPE tokenizer on lemmatized training texts
      4. Fit TF-IDF on training texts (for attention bias)
      5. Build DataLoaders
    """
    train_path = os.path.join(data_dir, "CONDA_train.csv")
    valid_path = os.path.join(data_dir, "CONDA_valid.csv")

    for p in [train_path, valid_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected data file not found: {p}")

    print("Loading CONDA splits...")
    train_texts_raw, train_labels = _load_split(train_path)
    val_texts_raw,   val_labels   = _load_split(valid_path)

    print(f"  Train : {len(train_texts_raw):,} samples")
    print(f"  Valid : {len(val_texts_raw):,} samples")

    # ── 1. Lemmatization ────────────────────────────────────────────────────
    print("\nLemmatizing...")
    lemmatizer       = Lemmatizer()
    train_texts_lem  = lemmatizer.batch_lemmatize(train_texts_raw)
    val_texts_lem    = lemmatizer.batch_lemmatize(val_texts_raw)

    # ── 2. BPE training ─────────────────────────────────────────────────────
    print("\nTraining BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=bpe_vocab_size)
    tokenizer.train(train_texts_lem)

    # ── 3. TF-IDF fitting ───────────────────────────────────────────────────
    print("\nFitting TF-IDF...")
    tfidf = TFIDFAttentionBias()
    tfidf.fit(train_texts_lem)

    # ── 4. Precompute TF-IDF vocab scores (one score per BPE token) ─────────
    #    We need a mapping from BPE token id → token string to look up IDF.
    if tokenizer._use_hf:
        id2token = {v: k for k, v in tokenizer._tokenizer.get_vocab().items()}
        token_strings = [id2token.get(i, "") for i in range(tokenizer.vocab_size_actual)]
    else:
        id2token = {v: k for k, v in tokenizer._char2idx.items()}
        token_strings = [id2token.get(i, "") for i in range(tokenizer.vocab_size_actual)]

    tfidf_vocab_scores = tfidf.precompute_vocab_scores(token_strings)  # (V,)
    ngram_ids_precomp  = None  # computed later inside model

    # ── 5. Class distribution & weights ─────────────────────────────────────
    from collections import Counter
    dist  = Counter(train_labels)
    inv   = {v: k for k, v in LABEL_MAP.items()}
    total = len(train_labels)
    print("\n  Train class distribution:")
    for idx in sorted(dist):
        print(f"    {inv[idx]} ({LABEL_NAMES[idx]}): {dist[idx]:,}")

    weights = torch.FloatTensor([total / (4 * dist.get(i, 1)) for i in range(4)])

    # ── 6. Build datasets & loaders ─────────────────────────────────────────
    def make_ds(texts, labels):
        return CONDADataset(texts, labels, tokenizer, max_len, tfidf_vocab_scores)

    train_ds = make_ds(train_texts_lem, train_labels)
    val_ds   = make_ds(val_texts_lem,   val_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    return (train_loader, val_loader, tokenizer, tfidf,
            tfidf_vocab_scores, weights, token_strings)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _EncoderLayer(nn.Module):
    """
    Single pre-norm transformer encoder layer with TF-IDF attention bias.

    The TF-IDF bias is injected as a (seq, seq) additive term in the attention
    logits.  We treat it as a *query-side* signal: position i attends more
    strongly to position j if token j has a high IDF score.

    tfidf_bias[i, j] = alpha * idf_score(token_j)

    This is computed externally (in the forward pass of the classifier) and
    passed in as `attn_bias`, which is then added to the raw QK^T scores
    before softmax.
    """

    def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.d_head    = d_model // num_heads

        # Manual QKV projection so we can inject attn_bias
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.ff1   = nn.Linear(d_model, dim_ff)
        self.ff2   = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

        # Learnable scale for the TF-IDF bias (starts near zero, learned)
        self.tfidf_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x: torch.Tensor,                            # (S, B, D)
        key_padding_mask: torch.Tensor = None,      # (B, S) bool
        attn_bias: torch.Tensor        = None,      # (B, 1, S) tfidf per key position
    ) -> torch.Tensor:
        S, B, D = x.shape
        H       = self.num_heads
        d       = self.d_head

        # Pre-norm
        x2 = self.norm1(x)                          # (S, B, D)
        x2 = x2.transpose(0, 1)                     # (B, S, D)

        # QKV
        Q = self.q_proj(x2).view(B, S, H, d).transpose(1, 2)  # (B,H,S,d)
        K = self.k_proj(x2).view(B, S, H, d).transpose(1, 2)
        V = self.v_proj(x2).view(B, S, H, d).transpose(1, 2)

        scale  = d ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale   # (B,H,S,S)

        # ── TF-IDF bias injection ────────────────────────────────────────
        # attn_bias: (B, 1, S)  →  broadcast to (B, H, S, S) by treating
        # the key dimension as the signal source.
        if attn_bias is not None:
            # attn_bias[b, 0, j] = IDF score of token j
            # Expand: (B, 1, 1, S) → (B, H, S, S)
            bias = attn_bias.unsqueeze(2) * self.tfidf_alpha   # (B,1,1,S)
            scores = scores + bias

        # Padding mask
        if key_padding_mask is not None:
            # (B, S) → (B, 1, 1, S)
            mask   = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        attn_w = torch.softmax(scores, dim=-1)
        attn_w = self.drop(attn_w)

        out = torch.matmul(attn_w, V)                           # (B,H,S,d)
        out = out.transpose(1, 2).contiguous().view(B, S, D)    # (B,S,D)
        out = self.o_proj(out).transpose(0, 1)                  # (S,B,D)

        x = x + self.drop(out)

        # Feed-forward
        x2 = self.norm2(x)
        x2 = self.ff2(self.drop(torch.relu(self.ff1(x2))))
        x  = x + self.drop(x2)
        return x


class TransformerEncoderClassifier(nn.Module):
    """
    Encoder-only transformer for 4-class intent classification.

    Embedding stack:
        token_emb(id)          — standard learned embedding
      + char_ngram_emb(token)  — FastText-style character n-gram embedding
      + pos_enc                — learned positional encoding

    Attention:
        Standard multi-head self-attention with injected TF-IDF bias.

    Pooling:
        [CLS] token at position 0 (prepended by BPE tokenizer).

    Input  : (batch, seq_len)  — BPE token ids
    Output : (batch, 4)        — raw logits
    """

    def __init__(
        self,
        vocab_size:   int,
        d_model:      int   = 128,
        num_heads:    int   = 4,
        num_layers:   int   = 2,
        dim_ff:       int   = 256,
        num_classes:  int   = 4,
        max_len:      int   = 64,
        dropout:      float = 0.1,
        pad_idx:      int   = 0,
        # Char n-gram params
        use_char_ngram:  bool = True,
        ngram_buckets:   int  = 20000,
        # TF-IDF params
        use_tfidf_bias:  bool = True,
    ):
        super().__init__()
        self.pad_idx        = pad_idx
        self.d_model        = d_model
        self.max_len        = max_len
        self.vocab_size     = vocab_size
        self.use_char_ngram = use_char_ngram
        self.use_tfidf_bias = use_tfidf_bias

        # ── Token embedding + positional encoding ────────────────────────
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        self.drop      = nn.Dropout(dropout)

        # ── Character n-gram embedding ───────────────────────────────────
        if use_char_ngram:
            self.char_ngram_emb = CharNgramEmbedding(
                num_buckets=ngram_buckets, d_model=d_model
            )
            # Buffer will be registered after tokenizer is built
            self.register_buffer("ngram_ids", None)

        # ── TF-IDF vocab scores buffer ───────────────────────────────────
        if use_tfidf_bias:
            self.register_buffer("tfidf_vocab_scores", None)

        # ── Encoder layers ───────────────────────────────────────────────
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

    def register_ngram_ids(self, ngram_ids: torch.Tensor):
        """Register precomputed n-gram id table as a buffer (no grad)."""
        self.register_buffer("ngram_ids", ngram_ids)

    def register_tfidf_scores(self, scores: torch.Tensor):
        """Register precomputed TF-IDF vocab scores as a buffer."""
        self.register_buffer("tfidf_vocab_scores", scores)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids : (batch, seq_len)
        returns   : (batch, num_classes) logits
        """
        key_padding_mask = (input_ids == self.pad_idx)     # (B, S)

        # ── Embedding ────────────────────────────────────────────────────
        x = self.embedding(input_ids)                      # (B, S, D)
        x = x + self.pos_enc[:, :x.size(1), :]

        # Add character n-gram signal
        if self.use_char_ngram and self.ngram_ids is not None:
            # Clamp ids to valid vocab range (safety for OOV edge cases)
            safe_ids = input_ids.clamp(0, self.ngram_ids.size(0) - 1)
            char_emb = self.char_ngram_emb(safe_ids, self.ngram_ids)  # (B,S,D)
            x = x + char_emb

        x = self.drop(x)

        # ── TF-IDF attention bias ─────────────────────────────────────────
        attn_bias = None
        if self.use_tfidf_bias and self.tfidf_vocab_scores is not None:
            # Gather IDF score for each token in the sequence
            safe_ids  = input_ids.clamp(0, self.tfidf_vocab_scores.size(0) - 1)
            attn_bias = self.tfidf_vocab_scores[safe_ids]              # (B, S)
            attn_bias = attn_bias.unsqueeze(1)                         # (B, 1, S)

        # ── Encoder ──────────────────────────────────────────────────────
        x = x.transpose(0, 1)                              # (S, B, D)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        x = self.norm(x)

        cls = x[0]                                         # [CLS] at position 0
        return self.classifier(self.drop(cls))             # (B, num_classes)


def build_model(
    tokenizer:          BPETokenizer,
    tfidf_vocab_scores: torch.Tensor,
    token_strings:      List[str],
    device:             torch.device,
    max_len:            int = 64,
) -> TransformerEncoderClassifier:
    """Instantiate model, register buffers, and move to device."""
    model = TransformerEncoderClassifier(
        vocab_size      = tokenizer.vocab_size_actual,
        d_model         = 128,
        num_heads       = 4,
        num_layers      = 2,
        dim_ff          = 256,
        num_classes     = 4,
        max_len         = max_len,
        dropout         = 0.1,
        pad_idx         = tokenizer.pad_id,
        use_char_ngram  = True,
        ngram_buckets   = 20000,
        use_tfidf_bias  = True,
    ).to(device)

    # Precompute and register char n-gram table
    ngram_ids = model.char_ngram_emb.precompute_token_ngram_ids(token_strings)
    model.register_ngram_ids(ngram_ids.to(device))

    # Register TF-IDF scores
    model.register_tfidf_scores(tfidf_vocab_scores.to(device))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__} | "
          f"Params: {n_params:,} | BPE vocab: {tokenizer.vocab_size_actual:,}")
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
    model.eval()
    total_loss             = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels    = labels.to(device)
            logits    = model(input_ids)
            loss      = criterion(logits, labels)
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
# Inference helpers
# ---------------------------------------------------------------------------

def _preprocess_for_inference(
    texts: List[str], lemmatizer: Lemmatizer
) -> List[str]:
    """Lemmatize inference texts before tokenizing."""
    return lemmatizer.batch_lemmatize(texts)


def predict(
    model:       nn.Module,
    texts:       List[str],
    tokenizer:   BPETokenizer,
    lemmatizer:  Lemmatizer,
    device:      torch.device,
    max_len:     int = 64,
    batch_size:  int = 64,
) -> np.ndarray:
    """Run inference on raw utterance strings → integer class predictions."""
    model.eval()
    texts_lem = _preprocess_for_inference(texts, lemmatizer)
    all_preds = []

    for i in range(0, len(texts_lem), batch_size):
        batch = texts_lem[i : i + batch_size]
        ids   = tokenizer.batch_encode(batch, max_len)
        t     = torch.tensor(ids, dtype=torch.long).to(device)
        with torch.no_grad():
            preds = model(t).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds)


def predict_proba(
    model:      nn.Module,
    texts:      List[str],
    tokenizer:  BPETokenizer,
    lemmatizer: Lemmatizer,
    device:     torch.device,
    max_len:    int = 64,
) -> np.ndarray:
    """Return softmax probability distributions over the 4 intent classes. Shape: (N, 4)"""
    model.eval()
    texts_lem = _preprocess_for_inference(texts, lemmatizer)
    ids       = tokenizer.batch_encode(texts_lem, max_len)
    t         = torch.tensor(ids, dtype=torch.long).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)
    return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    model:      nn.Module,
    tokenizer:  BPETokenizer,
    history:    Dict,
    metrics:    Dict,
    output_dir: str = "output",
    prefix:     str = "toxscan",
):
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

    # BPE tokenizer
    tok_path = os.path.join(output_dir, f"{prefix}_tokenizer.json")
    tokenizer.save(tok_path)

    # Metrics
    metrics_path = os.path.join(output_dir, f"{prefix}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
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


def load_model_and_tokenizer(
    device:     torch.device,
    output_dir: str = "output",
    prefix:     str = "toxscan",
) -> Tuple[Optional[nn.Module], Optional[BPETokenizer]]:
    """Load saved checkpoint + tokenizer if they exist."""
    ckpt_path = os.path.join(output_dir, f"{prefix}_model.pt")
    tok_path  = os.path.join(output_dir, f"{prefix}_tokenizer.json")

    if not os.path.exists(ckpt_path) or not os.path.exists(tok_path):
        return None, None

    print(f"Found existing checkpoint — loading from '{output_dir}/'")

    tokenizer = BPETokenizer()
    tokenizer.load(tok_path)

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = TransformerEncoderClassifier(
        vocab_size  = ckpt["vocab_size"],
        d_model     = ckpt["d_model"],
        num_classes = ckpt["num_classes"],
        max_len     = ckpt["max_len"],
        pad_idx     = tokenizer.pad_id,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    print("Model loaded successfully — skipping training.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ToxScan AI — CONDA intent classifier")
    parser.add_argument("--data_dir",      default="data",   help="Folder with train/valid CSV")
    parser.add_argument("--output_dir",    default="output", help="Where to save artifacts")
    parser.add_argument("--epochs",        type=int,   default=15)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=5e-4)
    parser.add_argument("--max_len",       type=int,   default=64)
    parser.add_argument("--patience",      type=int,   default=4)
    parser.add_argument("--bpe_vocab",     type=int,   default=8000,
                        help="BPE vocabulary size")
    args = parser.parse_args()

    print("=" * 65)
    print("ToxScan AI — Transformer Encoder for In-Game Toxicity Detection")
    print("  + BPE subword tokenization")
    print("  + spaCy lemmatization")
    print("  + Character n-gram embeddings")
    print("  + TF-IDF attention bias")
    print("=" * 65)

    set_seed(42)
    device   = get_device()
    metadata = get_task_metadata()
    print(f"\nTask   : {metadata['task_name']}")
    print(f"Classes: {metadata['label_map']}")

    # Always need lemmatizer for inference
    lemmatizer = Lemmatizer()

    # ── Load or Train ──────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(device, output_dir=args.output_dir)

    if model is None:
        print("\n── Data loading & preprocessing ─────────────────────────────")
        (train_loader, val_loader,
         tokenizer, tfidf,
         tfidf_vocab_scores, weights,
         token_strings) = make_dataloaders(
            data_dir      = args.data_dir,
            batch_size    = args.batch_size,
            max_len       = args.max_len,
            bpe_vocab_size= args.bpe_vocab,
        )

        print("\n── Model ────────────────────────────────────────────────────")
        model     = build_model(tokenizer, tfidf_vocab_scores,
                                token_strings, device, max_len=args.max_len)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
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

        save_artifacts(model, tokenizer, history,
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
        "ur so bad lmao get rekt noob",       # slang + abbreviations
        "i reported u already",               # typo-style shortening
        "tr4sh pl4yer uninstall the game",    # leet-speak
        "please report the feeder",           # implicit via action
    ]

    preds = predict(model, sample_utterances, tokenizer, lemmatizer,
                    device, max_len=args.max_len)
    probs = predict_proba(model, sample_utterances, tokenizer, lemmatizer,
                          device, max_len=args.max_len)

    print(f"\n{'Utterance':<45} {'Pred':>6}  {'Conf':>6}  Dist(E/I/A/O)")
    print("-" * 80)
    for utt, pred, prob in zip(sample_utterances, preds, probs):
        label = inv_label[int(pred)]
        conf  = prob[int(pred)]
        dist  = " ".join(f"{p:.2f}" for p in prob)
        print(f"{utt[:43]:<45} {label:>6}  {conf:>6.2%}  [{dist}]")

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
        user_input = input("Enter an utterance to evaluate (or 'q' to quit): ")
        if user_input.lower() == "q":
            break

        preds_i = predict(model, [user_input], tokenizer, lemmatizer,
                          device, max_len=args.max_len)
        probs_i = predict_proba(model, [user_input], tokenizer, lemmatizer,
                                device, max_len=args.max_len)

        label = inv_label[int(preds_i[0])]
        conf  = probs_i[0][int(preds_i[0])]
        dist  = " ".join(f"{LABEL_NAMES[i][0]}={probs_i[0][i]:.2f}" for i in range(4))

        print(f"\n  Prediction : {label} ({LABEL_NAMES[LABEL_MAP[label]]})")
        print(f"  Confidence : {conf:.2%}")
        print(f"  Distribution: [{dist}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())