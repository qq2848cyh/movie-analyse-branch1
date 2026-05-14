import os
import sys
import json
import pickle
import time
import numpy as np
from collections import Counter

import jieba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from gensim.models import Word2Vec

from .stopwords import get_dl_stop_words

try:
    from ..config import (
        VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, MAX_SEQ_LEN, BATCH_SIZE,
        CACHE_SENTIMENT_DIR, WORD2VEC_PATH, WORD2IDX_PATH, EMBED_MATRIX_PATH,
        PUNCT_INDICES_PATH, DL_MODEL_BINARY_PATH, DL_MODEL_FIVE_PATH,
        SENTIMENT_CURVES_DIR
    )
except (ImportError, ValueError):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import (
        VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, MAX_SEQ_LEN, BATCH_SIZE,
        CACHE_SENTIMENT_DIR, WORD2VEC_PATH, WORD2IDX_PATH, EMBED_MATRIX_PATH,
        PUNCT_INDICES_PATH, DL_MODEL_BINARY_PATH, DL_MODEL_FIVE_PATH,
        SENTIMENT_CURVES_DIR
    )

SENTIMENT_CACHE_DIR = CACHE_SENTIMENT_DIR

DL_STOP_WORDS = get_dl_stop_words()

PUNCT_CHARS = set("，。！？；：、…～—·,.:;!?\"'()[]{}<>/-_=+@#$%^&*|\\`~ \t\n\r")


def _is_punctuation(word):
    if len(word) == 1 and word in PUNCT_CHARS:
        return True
    return all(c in PUNCT_CHARS for c in word)


def tokenize_dl(text):
    words = jieba.lcut(str(text))
    return [
        w for w in words
        if len(w) >= 1
        and w not in DL_STOP_WORDS
    ]


def _build_vocab_and_word2vec(db_manager):
    if os.path.exists(WORD2IDX_PATH) and os.path.exists(WORD2VEC_PATH) and os.path.exists(EMBED_MATRIX_PATH):
        with open(WORD2IDX_PATH, "r", encoding="utf-8") as f:
            word2idx = json.load(f)
        embed_matrix = np.load(EMBED_MATRIX_PATH)
        if os.path.exists(PUNCT_INDICES_PATH):
            with open(PUNCT_INDICES_PATH, "r") as f:
                punct_indices = set(json.load(f))
        else:
            punct_indices = _build_punct_indices(word2idx)
            with open(PUNCT_INDICES_PATH, "w") as f:
                json.dump(list(punct_indices), f)
        print("[sentiment_dl] 词表与词向量从缓存加载 ✓", flush=True)
        return word2idx, embed_matrix, punct_indices

    c = db_manager._get_conn()
    rows = c.execute(
        "SELECT content FROM comments_valid "
        "WHERE content != '' "
        "ORDER BY RANDOM() LIMIT 100000"
    ).fetchall()

    print(f"[sentiment_dl] 构建词表，语料 {len(rows)} 条...", flush=True)
    sentences = [tokenize_dl(r["content"]) for r in rows]

    word_counts = Counter()
    for s in sentences:
        for w in s:
            word_counts[w] += 1

    top_words = word_counts.most_common(VOCAB_SIZE - 2)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for w, _ in top_words:
        word2idx[w] = len(word2idx)

    os.makedirs(SENTIMENT_CACHE_DIR, exist_ok=True)
    with open(WORD2IDX_PATH, "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False)
    print(f"[sentiment_dl] 词表已保存 ({len(word2idx)} 词)", flush=True)

    punct_indices = _build_punct_indices(word2idx)
    with open(PUNCT_INDICES_PATH, "w") as f:
        json.dump(list(punct_indices), f)
    print(f"[sentiment_dl] 标点索引已记录 ({len(punct_indices)} 个)", flush=True)

    print("[sentiment_dl] 训练 Word2Vec...", flush=True)
    w2v_model = Word2Vec(
        sentences,
        vector_size=EMBED_DIM,
        window=5,
        min_count=3,
        sg=1,
        workers=4,
    )
    w2v_model.save(WORD2VEC_PATH)
    print("[sentiment_dl] Word2Vec 已保存", flush=True)

    embed_matrix = np.zeros((VOCAB_SIZE, EMBED_DIM), dtype=np.float32)
    for word, idx in word2idx.items():
        if word in w2v_model.wv:
            embed_matrix[idx] = w2v_model.wv[word]
    np.save(EMBED_MATRIX_PATH, embed_matrix)

    return word2idx, embed_matrix, punct_indices


def _build_punct_indices(word2idx):
    punct_indices = set()
    for word, idx in word2idx.items():
        if _is_punctuation(word):
            punct_indices.add(idx)
    punct_indices.discard(0)
    punct_indices.discard(1)
    return punct_indices


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, num_classes=2,
                 max_seq_len=MAX_SEQ_LEN, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_dim = hidden_dim * 2

        self.attn_W = nn.Linear(self.lstm_dim, 64)
        self.attn_v = nn.Linear(64, 1)

        self.layer_norm = nn.LayerNorm(self.lstm_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.lstm_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _build_combined_mask(self, x, punct_indices, device):
        pad_mask = (x != 0)
        if punct_indices is not None and len(punct_indices) > 0:
            punct_mask = torch.ones_like(x, dtype=torch.bool, device=device)
            for pi in punct_indices:
                punct_mask = punct_mask & (x != pi)
            return pad_mask & punct_mask
        return pad_mask

    def forward(self, x, return_attention=False, punct_indices=None):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)

        scores = self.attn_v(self.attn_W(lstm_out)).squeeze(-1)

        mask = self._build_combined_mask(x, punct_indices, x.device)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=1)
        attn_weights = attn_weights * mask.float()

        mask_float = mask.float().unsqueeze(-1)
        attn_pooled = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)

        lstm_pooled = (lstm_out * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        combined = self.layer_norm(attn_pooled + lstm_pooled)

        out = self.dropout(combined)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)

        if return_attention:
            return out, attn_weights
        return out


def _evaluate_binary(model, dataloader, device, punct_indices):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x, punct_indices=punct_indices)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    return {
        "accuracy": round(float(accuracy_score(all_labels, all_preds)), 4),
        "precision": round(float(precision_score(all_labels, all_preds, average='binary')), 4),
        "recall": round(float(recall_score(all_labels, all_preds, average='binary')), 4),
        "f1": round(float(f1_score(all_labels, all_preds, average='binary')), 4),
    }


def _evaluate_five_class(model, dataloader, device, punct_indices):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x, punct_indices=punct_indices)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4])
    return {
        "accuracy": round(float(accuracy_score(all_labels, all_preds)), 4),
        "precision_macro": round(float(precision_score(all_labels, all_preds, average='macro')), 4),
        "recall_macro": round(float(recall_score(all_labels, all_preds, average='macro')), 4),
        "f1_macro": round(float(f1_score(all_labels, all_preds, average='macro')), 4),
        "confusion_matrix": cm.tolist(),
        "per_class_f1": [
            round(float(f), 4)
            for f in f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2, 3, 4])
        ],
    }


def train_bilstm_attn(db_manager, n_samples=15000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sentiment_dl] 使用设备: {device}", flush=True)

    word2idx, embed_matrix, punct_indices = _build_vocab_and_word2vec(db_manager)

    print(f"[sentiment_dl] 加载训练数据 {n_samples} 条...", flush=True)
    c = db_manager._get_conn()
    rows = c.execute(
        "SELECT content, rating FROM comments_valid "
        "WHERE content != '' AND rating IS NOT NULL "
        "ORDER BY CAST(rowid * 10007 % 100003 AS INTEGER) LIMIT ?",
        (n_samples,),
    ).fetchall()

    texts = [r["content"] for r in rows]
    ratings = [r["rating"] for r in rows]
    labels_binary = np.array([1 if r >= 4 else 0 for r in ratings])
    labels_five = np.array([int(r) - 1 for r in ratings])

    tokenized = [tokenize_dl(t) for t in texts]
    sequences = [
        [word2idx.get(w, 1) for w in seq]
        for seq in tokenized
    ]

    for i, seq in enumerate(sequences):
        if len(seq) > MAX_SEQ_LEN:
            sequences[i] = seq[:MAX_SEQ_LEN]
        else:
            sequences[i] = seq + [0] * (MAX_SEQ_LEN - len(seq))

    X = torch.tensor(sequences, dtype=torch.long)

    X_train, X_test, y_train, y_test, y5_train, y5_test = train_test_split(
        X, labels_binary, labels_five,
        test_size=0.2, random_state=42, stratify=labels_binary
    )

    train_ds = TensorDataset(X_train, torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(X_test, torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    train5_ds = TensorDataset(X_train, torch.tensor(y5_train, dtype=torch.long))
    test5_ds = TensorDataset(X_test, torch.tensor(y5_test, dtype=torch.long))
    train5_loader = DataLoader(train5_ds, batch_size=BATCH_SIZE, shuffle=True)
    test5_loader = DataLoader(test5_ds, batch_size=BATCH_SIZE)

    t0 = time.time()

    print("[sentiment_dl] 训练二分类 BiLSTM-Attention...", flush=True)
    model = BiLSTMAttention(vocab_size=VOCAB_SIZE, num_classes=2)
    model.embedding.weight.data.copy_(torch.from_numpy(embed_matrix))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_f1 = 0
    best_state = None
    patience = 0

    total_epochs = 30
    t_start = time.time()
    total_batches = len(train_loader)
    history_2cls = []
    for epoch in range(total_epochs):
        model.train()
        t_epoch = time.time()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x, punct_indices=punct_indices)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x, punct_indices=punct_indices)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch_y.numpy())
        val_f1 = f1_score(val_labels, val_preds, average='binary')
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(1 - val_f1)

        history_2cls.append({
            "epoch": epoch + 1,
            "val_f1": round(float(val_f1), 4),
            "val_acc": round(float(val_acc), 4),
            "epoch_time": int(time.time() - t_epoch),
        })

        e_elapsed = int(time.time() - t_epoch)
        total_elapsed = int(time.time() - t_start)
        if epoch > 0:
            eta = total_elapsed * (total_epochs - epoch) // epoch
        else:
            eta = 0
        tag = "↑" if val_f1 > best_f1 else f"·stop{patience+1}/5"
        pct = int((epoch + 1) / total_epochs * 20)
        bar = "█" * pct + "░" * (20 - pct)
        print(f"  [BiLSTM 2cls] [{bar}] epoch {epoch+1:2d}/{total_epochs} | "
              f"{total_batches}batches/{e_elapsed}s | "
              f"f1={val_f1:.4f} best={best_f1:.4f} {tag} | "
              f"elapsed {total_elapsed//60}m{total_elapsed%60:02d}s ETA {eta//60}m{eta%60:02d}s",
              flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 5:
            break

    model.load_state_dict(best_state)
    binary_metrics = _evaluate_binary(model, test_loader, device, punct_indices)
    print(f"[sentiment_dl] 二分类: acc={binary_metrics['accuracy']}, f1={binary_metrics['f1']}", flush=True)

    train_metrics_2cls = _evaluate_binary(model, train_loader, device, punct_indices)
    print(f"[sentiment_dl] 二分类训练集: acc={train_metrics_2cls['accuracy']}, f1={train_metrics_2cls['f1']}", flush=True)

    binary_state = best_state

    print("[sentiment_dl] 训练五分类 BiLSTM-Attention...", flush=True)
    model5 = BiLSTMAttention(vocab_size=VOCAB_SIZE, num_classes=5)
    model5.embedding.weight.data.copy_(torch.from_numpy(embed_matrix))
    model5.to(device)

    optimizer5 = optim.Adam(model5.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler5 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer5, mode='min', factor=0.5, patience=3
    )

    best_f1_5 = 0
    best_state_5 = None
    patience_5 = 0

    total5_batches = len(train5_loader)
    t_start5 = time.time()
    history_5cls = []
    for epoch in range(total_epochs):
        model5.train()
        t_epoch = time.time()
        for batch_x, batch_y in train5_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer5.zero_grad()
            logits = model5(batch_x, punct_indices=punct_indices)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer5.step()

        model5.eval()
        val_preds_5, val_labels_5 = [], []
        with torch.no_grad():
            for batch_x, batch_y in test5_loader:
                batch_x = batch_x.to(device)
                logits = model5(batch_x, punct_indices=punct_indices)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds_5.extend(preds)
                val_labels_5.extend(batch_y.numpy())
        val_f1_5 = f1_score(val_labels_5, val_preds_5, average='macro')
        val_acc_5 = accuracy_score(val_labels_5, val_preds_5)
        scheduler5.step(1 - val_f1_5)

        history_5cls.append({
            "epoch": epoch + 1,
            "val_f1": round(float(val_f1_5), 4),
            "val_acc": round(float(val_acc_5), 4),
            "epoch_time": int(time.time() - t_epoch),
        })

        e_elapsed = int(time.time() - t_epoch)
        total_elapsed = int(time.time() - t_start5)
        if epoch > 0:
            eta = total_elapsed * (total_epochs - epoch) // epoch
        else:
            eta = 0
        tag = "↑" if val_f1_5 > best_f1_5 else f"·stop{patience_5+1}/5"
        pct = int((epoch + 1) / total_epochs * 20)
        bar = "█" * pct + "░" * (20 - pct)
        print(f"  [BiLSTM 5cls] [{bar}] epoch {epoch+1:2d}/{total_epochs} | "
              f"{total5_batches}batches/{e_elapsed}s | "
              f"f1={val_f1_5:.4f} best={best_f1_5:.4f} {tag} | "
              f"elapsed {total_elapsed//60}m{total_elapsed%60:02d}s ETA {eta//60}m{eta%60:02d}s",
              flush=True)

        if val_f1_5 > best_f1_5:
            best_f1_5 = val_f1_5
            best_state_5 = {k: v.cpu().clone() for k, v in model5.state_dict().items()}
            patience_5 = 0
        else:
            patience_5 += 1
        if patience_5 >= 5:
            break

    model5.load_state_dict(best_state_5)
    five_metrics = _evaluate_five_class(model5, test5_loader, device, punct_indices)
    print(f"[sentiment_dl] 五分类: acc={five_metrics['accuracy']}, f1_macro={five_metrics['f1_macro']}", flush=True)

    train5_m = _evaluate_five_class(model5, train5_loader, device, punct_indices)
    print(f"[sentiment_dl] 五分类训练集: acc={train5_m['accuracy']}, f1_macro={train5_m['f1_macro']}", flush=True)

    training_time = round(time.time() - t0, 1)
    num_params = sum(p.numel() for p in model.parameters())

    torch.save(binary_state, DL_MODEL_BINARY_PATH)
    torch.save(best_state_5, DL_MODEL_FIVE_PATH)

    os.makedirs(SENTIMENT_CURVES_DIR, exist_ok=True)
    with open(os.path.join(SENTIMENT_CURVES_DIR, "dl_2cls_history.json"), "w", encoding="utf-8") as f:
        json.dump(history_2cls, f, ensure_ascii=False, indent=2)
    with open(os.path.join(SENTIMENT_CURVES_DIR, "dl_5cls_history.json"), "w", encoding="utf-8") as f:
        json.dump(history_5cls, f, ensure_ascii=False, indent=2)
    dl_curves = {
        "2cls": {"train": train_metrics_2cls, "test": binary_metrics},
        "5cls": {"train": {"accuracy": train5_m["accuracy"], "f1_macro": train5_m["f1_macro"]}, "test": {"accuracy": five_metrics["accuracy"], "f1_macro": five_metrics["f1_macro"]}},
    }
    with open(os.path.join(SENTIMENT_CURVES_DIR, "dl_train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dl_curves, f, ensure_ascii=False, indent=2)

    return {
        "binary": binary_metrics,
        "five_class": five_metrics,
        "training_time": training_time,
        "num_params": num_params,
    }


def _load_model_and_resources():
    with open(WORD2IDX_PATH, "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    embed_matrix = np.load(EMBED_MATRIX_PATH)

    if os.path.exists(PUNCT_INDICES_PATH):
        with open(PUNCT_INDICES_PATH, "r") as f:
            punct_indices = set(json.load(f))
    else:
        punct_indices = _build_punct_indices(word2idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTMAttention(vocab_size=VOCAB_SIZE, num_classes=2)
    model.embedding.weight.data.copy_(torch.from_numpy(embed_matrix))
    model.load_state_dict({k: v for k, v in torch.load(DL_MODEL_BINARY_PATH, map_location=device).items()})
    model.to(device)
    model.eval()

    model5 = BiLSTMAttention(vocab_size=VOCAB_SIZE, num_classes=5)
    model5.embedding.weight.data.copy_(torch.from_numpy(embed_matrix))
    model5.load_state_dict({k: v for k, v in torch.load(DL_MODEL_FIVE_PATH, map_location=device).items()})
    model5.to(device)
    model5.eval()

    return {
        "model": model,
        "model5": model5,
        "word2idx": word2idx,
        "punct_indices": punct_indices,
        "device": device,
    }


def get_attention_visualization(text, resources=None):
    if resources is None:
        resources = _load_model_and_resources()

    model = resources["model"]
    word2idx = resources["word2idx"]
    punct_indices = resources.get("punct_indices", set())
    device = resources["device"]

    raw_words = tokenize_dl(text)
    if not raw_words:
        return {"words": [], "weights": [], "error": "分词结果为空"}

    seq = [word2idx.get(w, 1) for w in raw_words]
    orig_len = len(seq)
    if len(seq) > MAX_SEQ_LEN:
        seq = seq[:MAX_SEQ_LEN]
        raw_words = raw_words[:MAX_SEQ_LEN]
        orig_len = len(seq)

    padded_seq = seq + [0] * (MAX_SEQ_LEN - len(seq))
    x = torch.tensor([padded_seq], dtype=torch.long).to(device)

    with torch.no_grad():
        _, attn_weights = model(x, return_attention=True, punct_indices=punct_indices)

    attn = attn_weights[0][:orig_len].cpu().numpy()

    total = float(attn.sum())
    if total > 0:
        attn = attn / total

    words = raw_words[:orig_len]
    weights = [round(float(v), 6) for v in attn]

    return {
        "words": words,
        "weights": weights,
        "max_weight": round(float(max(weights)) if weights else 0, 6),
        "min_weight": round(float(min(weights)) if weights else 0, 6),
    }
