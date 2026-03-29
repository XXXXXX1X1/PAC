# pip install gensim matplotlib

import re
from collections import Counter

import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


# =========================
# 1. Чтение текста
# =========================
file_name = "data/The_Fellowship_Of_The_Ring.txt"

with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

print("Файл загружен:", file_name)
print("Длина текста:", len(text))


# =========================
# 2. Подготовка данных
# =========================
def prepare_sentences(text):
    text = text.lower()
    text = text.replace("’", "'")
    parts = re.split(r"[.!?\n]+", text)

    sentences = []
    for part in parts:
        words = re.findall(r"[a-z']+", part)
        if len(words) >= 2:
            sentences.append(words)

    return sentences


sentences = prepare_sentences(text)
all_words = [w for sent in sentences for w in sent]
freq = Counter(all_words)

print("Количество предложений:", len(sentences))
print("Всего слов:", len(all_words))
print("Размер словаря:", len(freq))


# =========================
# 3. Логгер эпох
# =========================
class EpochLogger(CallbackAny2Vec):
    def __init__(self, name):
        self.name = name
        self.epoch = 0
        self.prev_loss = 0.0
        self.losses = []

    def on_epoch_begin(self, model):
        print(f"\n[{self.name}] Эпоха {self.epoch + 1} началась")

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        epoch_loss = loss - self.prev_loss
        self.prev_loss = loss
        self.losses.append(epoch_loss)

        print(f"[{self.name}] Эпоха {self.epoch + 1} завершена")
        print(f"[{self.name}] Loss за эпоху: {epoch_loss:.4f}")

        self.epoch += 1


# =========================
# 4. Skip-Gram
# =========================
skip_model = Word2Vec(
    vector_size=100,
    window=2,
    min_count=3,
    workers=1,
    sg=1,
    hs=0,
    negative=10,
    epochs=7,
    seed=42
)

skip_model.build_vocab(sentences)

skip_logger = EpochLogger("Skip-Gram")
skip_model.train(
    sentences,
    total_examples=skip_model.corpus_count,
    epochs=skip_model.epochs,
    compute_loss=True,
    callbacks=[skip_logger]
)

print("\nSkip-Gram обучена")
print("Размер словаря Skip-Gram:", len(skip_model.wv))


# =========================
# 5. Top-5 похожих слов
# =========================
def print_top5_similar(model, word):
    word = word.lower()

    if word not in model.wv:
        print(f"\nСлова '{word}' нет в словаре")
        return

    print(f"\nТоп-5 слов для '{word}':")
    for near_word, score in model.wv.most_similar(word, topn=5):
        print(f"{near_word:15s} {score:.4f}")


test_words = [
    "ring",
    "frodo",
    "gandalf",
    "shire",
    "sam",
    "bilbo",
    "mordor",
    "road",
    "hobbit",
    "black"
]

for word in test_words:
    print_top5_similar(skip_model, word)


# =========================
# 6. CBOW
# =========================
cbow_model = Word2Vec(
    vector_size=100,
    window=2,
    min_count=3,
    workers=1,
    sg=0,
    hs=0,
    negative=10,
    epochs=7,
    seed=42
)

cbow_model.build_vocab(sentences)

cbow_logger = EpochLogger("CBOW")
cbow_model.train(
    sentences,
    total_examples=cbow_model.corpus_count,
    epochs=cbow_model.epochs,
    compute_loss=True,
    callbacks=[cbow_logger]
)

print("\nCBOW обучена")
print("Размер словаря CBOW:", len(cbow_model.wv))


# =========================
# 7. Предсказание слова по контексту
# =========================
def predict_target_by_context(model, context_words, topn=5):
    context_words = [w.lower() for w in context_words]
    context_words = [w for w in context_words if w in model.wv.key_to_index]

    if not context_words:
        print("\nНет слов из контекста в словаре модели")
        return

    print("\nКонтекст:", context_words)

    result = model.predict_output_word(context_words, topn=topn)

    if result is None:
        print("Модель не смогла сделать предсказание")
        return

    print("Предсказание целевого слова:")
    for word, score in result:
        print(f"{word:15s} {score:.4f}")


context_examples = [
    ["frodo", "sam", "shire"],
    ["ring", "dark", "lord"],
    ["gandalf", "wizard", "fire"],
    ["black", "riders", "road"],
    ["bilbo", "baggins", "hobbit"],
    ["mountains", "snow", "wind"],
    ["elves", "forest", "night"],
    ["river", "boat", "water"],
    ["sword", "enemy", "battle"],
    ["road", "journey", "danger"]
]

for ctx in context_examples:
    predict_target_by_context(cbow_model, ctx)


# =========================
# 8. Предсказание пропущенного слова в предложении
# =========================
def predict_missing_word_from_sentence(model, sentence, missing_index, topn=5):
    words = re.findall(r"[a-z']+", sentence.lower())

    if missing_index < 0 or missing_index >= len(words):
        print("\nНеверный индекс пропущенного слова")
        return

    real_target = words[missing_index]
    context = words[:missing_index] + words[missing_index + 1:]
    context = [w for w in context if w in model.wv.key_to_index]

    if not context:
        print("\nКонтекст пустой после фильтрации")
        return

    print("\nИсходное предложение:", " ".join(words))
    print("Настоящее целевое слово:", real_target)
    print("Контекст без этого слова:", context)

    result = model.predict_output_word(context, topn=topn)

    if result is None:
        print("Модель не смогла сделать предсказание")
        return

    print("Топ предсказаний:")
    for word, score in result:
        print(f"{word:15s} {score:.4f}")


sentence_examples = [
    ("frodo and sam returned to the shire", 5),
    ("bilbo found the ring in the dark", 3),
    ("gandalf was a wise wizard", 3),
    ("the black riders followed the road", 2),
    ("the hobbit carried the ring", 3),
    ("sam stayed with frodo", 2),
    ("the company went through moria", 4),
    ("the river carried the boat", 2),
    ("the elves came in the night", 1),
    ("the road went ever on", 1)
]

for sent, idx in sentence_examples:
    predict_missing_word_from_sentence(cbow_model, sent, idx)


# =========================
# 9. График loss по эпохам
# =========================
skip_epochs = list(range(1, len(skip_logger.losses) + 1))
cbow_epochs = list(range(1, len(cbow_logger.losses) + 1))

plt.figure(figsize=(10, 5))
plt.plot(skip_epochs, skip_logger.losses, marker="o", label="Skip-Gram")
plt.plot(cbow_epochs, cbow_logger.losses, marker="o", label="CBOW")
plt.title("Loss по эпохам")
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================
# 10. График 20 самых частых слов
# =========================
top_n = 20
top_words = freq.most_common(top_n)

words = [w for w, c in top_words]
counts = [c for w, c in top_words]

plt.figure(figsize=(14, 6))
plt.bar(words, counts)
plt.title("20 самых часто встречающихся слов")
plt.xlabel("Слова")
plt.ylabel("Частота")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =========================
# 11. Сохранение моделей
# =========================
skip_model.save("skipgram_lotr.model")
cbow_model.save("cbow_lotr.model")

print("\nМодели сохранены")