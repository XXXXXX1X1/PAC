import re
import random
from pathlib import Path
from collections import Counter

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


WINDOW = 3
WINDOW_CBOW = 2
MAX_VOCAB = 200000
MIN_COUNT = 2
EMBEDDING_DIM = 200
BATCH_SIZE = 1024
EPOCHS_SG = 10
EPOCHS_CBOW = 10
K_NEG = 5
LR = 2e-3

SG_MODE = False
CBOW_MODE = False


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
    text = re.sub(r"[ \t]+", " ", text)
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
    id2word = {index: word for index, word in enumerate(vocab)}
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
        length = len(sentence)
        for index, center in enumerate(sentence):
            left = max(0, index - window)
            right = min(length, index + window + 1)
            for j in range(left, right):
                if j != index:
                    pairs.append((center, sentence[j]))
    return np.asarray(pairs, dtype=np.int64)

class SkipGramPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = torch.from_numpy(pairs).long()
    def __len__(self):
        return self.pairs.shape[0]
    def __getitem__(self, index):
        return self.pairs[index, 0], self.pairs[index, 1]

# =======================
# CBOW DATA
# =======================
class CBOWDataset(Dataset):
    def __init__(self,  sentences_as_ids, window_size = 2):
        self.sentences_as_ids = sentences_as_ids
        self.window_size = window_size
        self.samples_index = []

        for sentence_index, sentence_ids in enumerate(sentences_as_ids):
            min_length = 2 * window_size + 1

            if len(sentence_ids) < min_length:
                continue

            for target_position in range(window_size, len(sentence_ids) - window_size):
                self.samples_index.append((sentence_index, target_position))

    def __len__(self):
        return  len(self.samples_index)

    def __getitem__(self, sample_index):
        sentence_index, target_position = self.samples_index[sample_index]
        sentence_ids = self.sentences_as_ids[sentence_index]
        window_size = self.window_size

        left_context = sentence_ids[target_position - window_size:target_position]
        right_context = sentence_ids[target_position + 1:target_position + window_size + 1]

        context_ids = left_context + right_context
        target_id = sentence_ids[target_position]

        return (torch.tensor(context_ids, dtype=torch.long),
                torch.tensor(target_id, dtype=torch.long)
        )

# =============================
# MODELS
# =============================
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()

        self.in_embed = nn.Embedding(vocab_size, emb_dim)
        self.out_embed = nn.Embedding(vocab_size, emb_dim)

        nn.init.uniform_(self.in_embed.weight, -0.5 / emb_dim, 0.5 / emb_dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center_ids, pos_ids, neg_ids):
        center_vectors = self.in_embed(center_ids)
        positive_vectors = self.out_embed(pos_ids)
        negative_vectors = self.out_embed(neg_ids)

        positive_score = torch.sum(center_vectors * positive_vectors, dim=1)
        positive_loss = F.logsigmoid(positive_score)

        negative_score = torch.bmm(negative_vectors, center_vectors.unsqueeze(2)).squeeze(2)

        negative_loss = F.logsigmoid(-negative_score).sum(dim=1)

        return -(positive_loss + negative_loss).mean()


class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_word_ids):
        context_embeddings = self.word_embeddings(context_word_ids)
        context_mean_vector = context_embeddings.mean(dim=1)
        logits = self.output_layer(context_mean_vector)
        return logits

# ======================
# SAVE / LOAD
# ======================
def save_skipgram(model, path, vocab, word2id, id2word, vocab_size, embedding_dim):
    torch.save({
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "word2id": word2id,
        "id2word": id2word
    }, path)

def load_skipgram(path, device):
    checkpoint = torch.load(path, map_location=device)

    model = SkipGramNegSampling(
        checkpoint["vocab_size"],
        checkpoint["embedding_dim"]
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, checkpoint["vocab"], checkpoint["word2id"], checkpoint['id2word']

def save_cbow(model, path, vocab, word2id, id2word, vocab_size, embedding_dim):
    torch.save(
        {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "word2id": word2id,
            "id2word": id2word
        }, path)

def load_cbow(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = CBOW(
        checkpoint["vocab_size"],
        checkpoint["embedding_dim"]
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, checkpoint["vocab"], checkpoint["word2id"], checkpoint["id2word"]

def train_skipgram(skipgram_dataloader, negative_probs_device, vocab_size, vocab, word2id, id2word):
    if SG_MODE:
        model = SkipGramNegSampling(vocab_size, EMBEDDING_DIM).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(1, EPOCHS_SG + 1):
            model.train()
            total_loss = 0.0
            steps = 0

            for center_ids, positive_ids in skipgram_dataloader:
                center_ids = center_ids.to(device).long()
                positive_ids=positive_ids.to(device).long()

                negative_ids = torch.multinomial(negative_probs_device,
                                                 num_samples=center_ids.shape[0] * K_NEG,
                                                 replacement=True
                                                 ).view(center_ids.shape[0], K_NEG)
                optimizer.zero_grad(set_to_none=True)

                loss = model(center_ids, positive_ids, negative_ids)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                steps += 1
            epoch_loss = total_loss / steps if steps > 0 else 0.0
            print(f"[SkipGram] epoch {epoch}/{EPOCHS_SG} | loss = {epoch_loss:.4f}")

        save_skipgram(
            model,
            SG_WEIGHTS,
            vocab,
            word2id,
            id2word,
            vocab_size,
            EMBEDDING_DIM
        )
    else:
        model, vocab, word2id, id2word = load_skipgram(SG_WEIGHTS, device)
        print("Веса загружены")

    return model, vocab, word2id, id2word

def train_cbow(cbow_dataloader, vocab_size, vocab, word2id, id2word):
    if CBOW_MODE:
        model = CBOW(vocab_size, EMBEDDING_DIM).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(1, EPOCHS_SG + 1):
            model.train()
            totall_loss = 0.0
            steps = 0

            for context_ids, target_ids in cbow_dataloader:
                context_ids = context_ids.to(device).long()
                target_ids = target_ids.to(device).long()

                optimizer.zero_grad(set_to_none=True)

                logist = model(context_ids)
                loss = F.cross_entropy(logist, target_ids)

                loss.backward()
                optimizer.step()

                totall_loss += float(loss.item())
                steps += 1
            epoch_loss = totall_loss / steps if steps > 0 else 0
            print(f"[CBOW] epoch {epoch}/{EPOCHS_CBOW} | loss = {epoch_loss: .4f}")

        save_cbow(
            model,
            CBOW_WEIGHTS,
            vocab,
            word2id,
            id2word,
            vocab_size,
            EMBEDDING_DIM
        )
    else:
        model, vocab, word2id, id2word = load_cbow(CBOW_WEIGHTS, device)
        print("Веса загружены")
    return model, vocab, word2id, id2word

def topk_cosine_torch(word, model, word2id, id2word, topk=5):
    word = word.lower()

    if word not in word2id:
        return f"Слова '{word}' нет в словаре"

    embeddings = model.in_embed.weight.detach()
    embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-12)

    word_id = word2id[word]
    word_vector = embeddings[word_id]

    similarities = embeddings @ word_vector

    similarities[word_id] = -1.0
    if "<UNK>" in word2id:
        similarities[word2id["<UNK>"]] = -1.0

    best_ids = torch.topk(similarities, k=topk).indices.tolist()

    result = []
    for idx in best_ids:
        result.append((id2word[idx], float(similarities[idx])))

    return result

def predict_cbow(context_words, cbow_model, word2id, id2word, topk=5):
    context_ids = [word2id.get(word.lower(), word2id["<UNK>"]) for word in context_words]
    x = torch.tensor([context_ids], dtype=torch.long, device=device)

    cbow_model.eval()
    with torch.no_grad():
        logits = cbow_model(x)
        probabilities = torch.softmax(logits, dim=1)
        values, indices = torch.topk(probabilities, k=topk, dim=1)

    result = []
    for score, idx in zip(values[0].cpu().tolist(), indices[0].cpu().tolist()):
        result.append((id2word[idx], float(score)))

    return result
def main():
    text = read_all_txt(DATA_FILE)

    sentences = split_sentence(text)
    print("Количество предложений:", len(sentences))

    tokenized_sentences = [tokens for tokens in (tokenize(sentence) for sentence in sentences) if len(tokens) >= 2]
    total_tokens = sum(len(sentence) for sentence in tokenized_sentences)
    print("Количество токенов: ", total_tokens)


    vocab, word2id, id2word, counter = build_vocab(
        tokenized_sentences,
        max_vocab=MAX_VOCAB,
        min_count=MIN_COUNT
    )
    print("Размер словаря: ", len(vocab))
    print("Топ 10 слов: ", counter.most_common(10))

    unk_id = word2id["<UNK>"]

    encoded_sentences = []
    for sentence in tokenized_sentences:
        encoded_sentence = [word2id.get(word, unk_id) for word in sentence]
        encoded_sentences.append(encoded_sentence)
    total_encoded_tokens = sum(len(sentence) for sentence in encoded_sentences)
    unk_tokens = sum(1 for sentence in encoded_sentences for token_id in sentence if token_id == unk_id)

    print(unk_tokens)

    encoded_sentences_subsampled = subsample(encoded_sentences, t=1e-4)

    skipgram_pairs = make_skipgram_pairs(encoded_sentences_subsampled, window=WINDOW)

    skipgram_dataset = SkipGramPairDataset(skipgram_pairs)
    skipgram_dataloader = DataLoader(
        skipgram_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    vocab_size = len(vocab)

    id_counts = torch.zeros(vocab_size, dtype=torch.float32)
    for sentence in encoded_sentences:
        for token_id in sentence:
            id_counts[token_id] += 1

    negative_weights = id_counts.pow(0.75)
    negative_weights[unk_id] = 0.0
    negative_probs = negative_weights / negative_weights.sum()
    negative_probs_device = negative_probs.to(device)



    min_cbow_length = 2 * WINDOW_CBOW + 1
    cbow_sentences = [sentence for sentence in encoded_sentences if len(sentence) >= min_cbow_length]

    cbow_dataset = CBOWDataset(cbow_sentences, window_size=WINDOW_CBOW)
    cbow_dataloader = DataLoader(
        cbow_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    skipgram_model, vocab, word2id, id2word = train_skipgram(
        skipgram_dataloader,
        negative_probs_device,
        vocab_size,
        vocab,
        word2id,
        id2word
    )

    print("\n=== Skip-gram похожие слова ===")

    print("sam:")
    print(topk_cosine_torch("sam", skipgram_model, word2id, id2word, topk=5))

    print("\nfrodo:")
    print(topk_cosine_torch("frodo", skipgram_model, word2id, id2word, topk=5))

    print("\ngandalf:")
    print(topk_cosine_torch("gandalf", skipgram_model, word2id, id2word, topk=5))

    print("\nring:")
    print(topk_cosine_torch("ring", skipgram_model, word2id, id2word, topk=5))

    print("\nshire:")
    print(topk_cosine_torch("shire", skipgram_model, word2id, id2word, topk=5))
    cbow_model, vocab, word2id, id2word = train_cbow(
        cbow_dataloader,
        vocab_size,
        vocab,
        word2id,
        id2word
    )

    print("\n=== CBOW предсказание слова по контексту ===")

    print("\nContext: ['frodo', 'and', 'sam', 'went']")
    print(predict_cbow(["frodo", "and", "sam", "went"], cbow_model, word2id, id2word, topk=5))

    print("\nContext: ['the', 'one', 'ring', 'to']")
    print(predict_cbow(["the", "one", "ring", "to"], cbow_model, word2id, id2word, topk=5))

    print("\nContext: ['in', 'the', 'shire', 'lived']")
    print(predict_cbow(["in", "the", "shire", "lived"], cbow_model, word2id, id2word, topk=5))
if __name__ == "__main__":
    main()