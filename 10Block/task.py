# =====================================
# MNIST + average_digit + перцептрон + t-SNE
# Задание 0–7
# =====================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score
from sklearn.manifold import TSNE

import torchvision

plt.rcParams["figure.figsize"] = (6, 6)

# ----------------------------------------------------
# Пункт 0. Загрузка MNIST
# ----------------------------------------------------
train_dataset = torchvision.datasets.MNIST(
    root="./MNIST",
    train=True,
    download=True,
)
test_dataset = torchvision.datasets.MNIST(
    root="./MNIST",
    train=False,
    download=True,
)

print("=== 0. Данные ===")
print("train size:", len(train_dataset))
print("test  size:", len(test_dataset))

# ----------------------------------------------------
# Пункт 1. Перевод в (N, 784) + нормализация
# ----------------------------------------------------

X_train = train_dataset.data.view(-1, 28 * 28).numpy().astype(np.float32) / 255.0
y_train = train_dataset.targets.numpy().astype(np.int64)

X_test = test_dataset.data.view(-1, 28 * 28).numpy().astype(np.float32) / 255.0
y_test = test_dataset.targets.numpy().astype(np.int64)

print("\n=== 1. Формы массивов ===")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)

# ----------------------------------------------------
# Пункт 1  average_digit для каждой цифры 0–9
# ----------------------------------------------------
def average_digit(X, y, digit):
    """Среднее изображение (вектор 784) для заданной цифры digit."""
    mask = (y == digit)
    return X[mask].mean(axis=0)

# Матрица весов: 10 строк (классы), 784 столбца (пиксели)
avg_digits = np.stack([average_digit(X_train, y_train, d) for d in range(10)])  # (10, 784)

# Визуализация средних цифр (для отчёта)
fig_avg, axes = plt.subplots(2, 5, figsize=(8, 4))
for d in range(10):
    ax = axes[d // 5, d % 5]
    ax.imshow(avg_digits[d].reshape(28, 28), cmap="gray")
    ax.set_title(str(d))
    ax.axis("off")
fig_avg.suptitle("Средние изображения цифр 0–9 (average_digit)", y=0.98)
plt.tight_layout()

# ----------------------------------------------------
# Пункт 2. 10 бинарных классификаторов
# ----------------------------------------------------
biases = np.zeros(10, dtype=np.float32)

for d in range(10):
    w = avg_digits[d]             # веса для цифры d, размер 784
    scores = X_train @ w          # скалярные произведения для всех train-изображений (N,)

    pos_scores = scores[y_train == d]   # скоры для "правильного" класса d
    neg_scores = scores[y_train != d]   # скоры для остальных цифр

    # Порог ставим посередине между средними pos/neg → bias = -threshold
    thr = 0.5 * (pos_scores.mean() + neg_scores.mean())
    biases[d] = -thr

def predict_binary_digit(x_vec, d):
    """
    Бинарный классификатор для цифры d.
    Возвращает 1, если (w_d · x + b_d) >= 0, иначе 0.
    """
    w = avg_digits[d]
    b = biases[d]
    score = np.dot(w, x_vec) + b
    return 1 if score >= 0 else 0

# ----------------------------------------------------
# Пункт 3. Точность каждого бинарного классификатора на тесте
# ----------------------------------------------------
print("\n=== 2. Точность бинарных классификаторов (one-vs-all) на тесте ===")
digit_accuracies = []
for d in range(10):
    correct = 0
    total = len(X_test)
    for i in range(total):
        x = X_test[i]
        y_true = 1 if y_test[i] == d else 0
        y_pred = predict_binary_digit(x, d)
        if y_true == y_pred:
            correct += 1
    acc = correct / total
    digit_accuracies.append(acc)
    print(f"Классификатор цифры {d}: accuracy = {acc:.4f}")

# ----------------------------------------------------
# Пункт 4. Объединённая модель → вектор размера 10
# ----------------------------------------------------
def model_predict_vector(x_vec):
    return np.array([predict_binary_digit(x_vec, d) for d in range(10)], dtype=int)

def model_logits(x_vec):
    return avg_digits @ x_vec + biases  # (10,)

def model_predict_class_from_vector(x_vec):
    v = model_predict_vector(x_vec)   # вектор 0/1 по заданию
    s = model_logits(x_vec)

    ones = np.where(v == 1)[0]

    if len(ones) == 1:
        return int(ones[0])

    if len(ones) > 1:
        best_idx_in_ones = np.argmax(s[ones])
        return int(ones[best_idx_in_ones])

    return int(np.argmax(s))

# ----------------------------------------------------
# Пункт 5.
# ----------------------------------------------------
# Строим предсказанные классы для всех тестовых изображений
y_pred_test = np.array([model_predict_class_from_vector(x) for x in X_test])

# Accuracy по классам (процент правильно распознанных цифр)
overall_acc = np.mean(y_pred_test == y_test)


precision = precision_score(y_test, y_pred_test, average="macro")
recall = recall_score(y_test, y_pred_test, average="macro")

print("\n=== 3. Качество многоклассовой модели (10 классов) на тесте ===")
print(f"Accuracy          : {overall_acc:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")

# ----------------------------------------------------
# Пункт 6. t-SNE по сырым данным (784-мерные векторы пикселей)
# ----------------------------------------------------
# Берём по 30 изображений каждого класса из теста
indices_per_class = {d: [] for d in range(10)}
for i, label in enumerate(y_test):
    if len(indices_per_class[label]) < 30:
        indices_per_class[label].append(i)
    if all(len(v) == 30 for v in indices_per_class.values()):
        break

selected_indices = np.concatenate([indices_per_class[d] for d in range(10)])  # 300 штук
X_raw = X_test[selected_indices]  # (300, 784)
y_raw = y_test[selected_indices]  # (300,)

print("\nЗапускаем t-SNE по сырым пикселям (784)...")
tsne_raw = TSNE(
    n_components=2,
    init="random",
    learning_rate="auto",
    random_state=42,
)
X_raw_2d = tsne_raw.fit_transform(X_raw)

fig_tsne_raw = plt.figure(figsize=(7, 7))
for d in range(10):
    mask = (y_raw == d)
    plt.scatter(X_raw_2d[mask, 0], X_raw_2d[mask, 1], s=10, label=str(d))
plt.legend()
plt.title("t-SNE по сырым пикселям (784)")

# ----------------------------------------------------
# Пункт 7. t-SNE по логитам модели (10-мерные вектора)
# ----------------------------------------------------
print("Запускаем t-SNE по логитам модели (10)...")
logits = np.array([model_logits(X_test[i]) for i in selected_indices])  # (300, 10)

tsne_logits = TSNE(
    n_components=2,
    init="random",
    learning_rate="auto",
    random_state=42,
)
logits_2d = tsne_logits.fit_transform(logits)

fig_tsne_logits = plt.figure(figsize=(7, 7))
for d in range(10):
    mask = (y_raw == d)
    plt.scatter(logits_2d[mask, 0], logits_2d[mask, 1], s=10, label=str(d))
plt.legend()
plt.title("t-SNE по выходам модели (логиты, 10)")

# Показываем все фигуры разом
plt.show()

