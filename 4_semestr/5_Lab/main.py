import os
import re
import random
from pathlib import Path
from collections import Counter
from winreg import KEY_NOTIFY

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =======================
# Config
# =======================

seed = 42

DATA_FILE = Path('data') / "The_Fellowship_Of_The_Ring.txt"
WEIGHTS_DIR = Path('weights')
WEIGHTS_DIR.mkdir(exist_ok=True)

SG_WEIGHTS = WEIGHTS_DIR / "skipgram_weights.pt"
CBOW_WEIGHTS = WEIGHTS_DIR / "cbow_weights.pt"

WINDOW = 2
WIND0W_CBOW = 2
MAX_VOCAB = 200000
MIN_COUNT = 2
EMBEDDING_DIM = 200
BATCH_SIZE = 1024
EPOCHS_SG = 10
EPOCHS_CBOW = 10
K_NEG = 5

SG_MODE = True
CBOW_MODE = True


# =======================
# SEED/DEVICE
# =======================
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all(seed)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)

# ========================
# TEXT
# ========================
def read_all_txt(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8", errors="ignore").lower()
    print(len(text))
    return text


WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

def split_sentence(text: str) -> list[str]:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ /t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if any(ch in text for ch in [".", "?", "!"]):
        parts = re.split(r"[.?!]", text)
        parts = [p.strip() for p in parts if p.strip()]
        if parts:
            return parts

    lines = []
    for line in text.split("\n"):
        string = line.strip()
        if string:
            lines.append(string)

    return lines

# разбиваем стоку на токены
def tokenize(string: str) -> list[str]:
    words = WORD.findall(string)
    return words

# =====================
# VOCAB
# =====================
# составляем словарь слишком редкие слова не берем
def build_vocab(tokenized_sentences, max_vocab=MAX_VOCAB, min_count=MIN_COUNT):
    counter = Counter(token for sentence in tokenized_sentences for token in sentence)
    items = [(word, count) for word, count in counter.items() if count >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)

    vocab = ["<UNK>"] + [word for word, _ in items[:max(0, max_vocab - 1)]]
    word2id = {word: index for index, word in enumerate(vocab)}
    id2word = {word: index for index, word in enumerate(vocab)}
    return vocab, word2id, id2word, counter

# уменьшаем количество слишком частых слов
def subsample(encoder_sentences, t=1e-3):
    counter = Counter(word for s in encoder_sentences for word in s)
    total = sum(counter.values())
    freqs = {w:c / total for w, c in counter.items()}

    out = []
    for sentence in encoder_sentences:
        ns = []
        for word in sentence:
            freq = freqs[word]
            p_drop = max(0, 1 - (t / freq) ** 0.5)
            if random.random() > p_drop:
                ns.append(word)
        if len(ns) > 2:
            out.append(ns)
    return out

def make_skipgram_pairs(encoder_sentences, window = WINDOW):
    pairs = []
    for sentence in encoder_sentences:
        lenght = len(sentence)
        for index, center in enumerate(sentence):
            left = max(0, index - window)
            right = min(lenght, index + window + 1)
            for j in range(left, right):
                if j != 1:
                    pairs.append((center, sentence[j]))
    return np.asarray(pairs, dtype=np.int64)

class SkipGramPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = torch.from_numpy(pairs).long()
    def __len__(self):
        return self.pairs.shape[0]
    def __getitem__(self, index):
        return self.pairs[index, 0], self.pairs[index, 1]








def main():
    text = read_all_txt(DATA_FILE)






if __name__ == "__main__":
    main()