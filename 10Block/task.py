# =====================================
# MNIST + average_digit + перцептрон + t-SNE
# =====================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score
from sklearn.manifold import TSNE

import torchvision

plt.rcParams["figure.figsize"] = (6, 6)


# ------------------------------
# 0. Загрузка MNIST
# ------------------------------
transform = torchvision.transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="./MNIST",
    train=True,
    transform=transform,
    download=True,
)

test_dataset = torchvision.datasets.MNIST(
    root="./MNIST",
    train=False,
    transform=transform,
    download=True,
)

print("=== 0. Данные ===")
print("train size:", len(train_dataset))
print("test  size:", len(test_dataset))


# ------------------------------
# 1. Перевод датасета в массивы (N, 784) и метки (N,)
# ------------------------------
def dataset_to_arrays(dataset):
    X = []
    y = []
    for img, label in dataset:
        arr = img[0].numpy().reshape(-1)  # (784,)
        X.append(arr)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


X_train, y_train = dataset_to_arrays(train_dataset)
X_test, y_test = dataset_to_arrays(test_dataset)

print("\n=== 1. Формы массивов ===")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)


# ------------------------------
# 2. average_digit для каждой цифры 0–9
# ------------------------------
def average_digit(X, y, digit):
    mask = (y == digit)
    return X[mask].mean(axis=0)


avg_digits = np.stack([average_digit(X_train, y_train, d) for d in range(10)])  # (10, 784)

# Визуализация средних цифр (наглядность по заданию)
fig_avg, axes = plt.subplots(2, 5, figsize=(8, 4))
for d in range(10):
    ax = axes[d // 5, d % 5]
    ax.imshow(avg_digits[d].reshape(28, 28), cmap="gray")
    ax.set_title(str(d))
    ax.axis("off")
fig_avg.suptitle("Средние изображения цифр 0–9 (average_digit)", y=0.98)
plt.tight_layout()


# ------------------------------
# 3. Десять бинарных классификаторов (one-vs-all) + bias
# ------------------------------
biases = np.zeros(10, dtype=np.float32)

for d in range(10):
    w = avg_digits[d]                # (784,)
    scores = X_train @ w             # (N,)

    pos_scores = scores[y_train == d]
    neg_scores = scores[y_train != d]

    thr = 0.5 * (pos_scores.mean() + neg_scores.mean())
    biases[d] = -thr                 # bias = -threshold


def predict_binary_digit(x_vec, d):
    w = avg_digits[d]
    b = biases[d]
    score = np.dot(w, x_vec) + b
    return 1 if score >= 0 else 0


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


# ------------------------------
# 4. Объединённая модель (логиты 10)
# ------------------------------
def model_logits(x_vec):
    # avg_digits: (10, 784), biases: (10,)
    return avg_digits @ x_vec + biases  # (10,)


def model_predict_class(x_vec):
    scores = model_logits(x_vec)
    return int(np.argmax(scores))


def model_predict_one_hot(x_vec):
    scores = model_logits(x_vec)
    out = np.zeros(10, dtype=int)
    out[np.argmax(scores)] = 1
    return out


# ------------------------------
# 5. Accuracy, precision, recall для многоклассовой модели
# ------------------------------
y_pred_test = np.array([model_predict_class(x) for x in X_test])

overall_acc = np.mean(y_pred_test == y_test)
precision_macro = precision_score(y_test, y_pred_test, average="macro")
recall_macro = recall_score(y_test, y_pred_test, average="macro")

print("\n=== 3. Качество многоклассовой модели (10 классов) на тесте ===")
print(f"Accuracy          : {overall_acc:.4f}")
print(f"Precision (macro) : {precision_macro:.4f}")
print(f"Recall    (macro) : {recall_macro:.4f}")


# ------------------------------
# 6. t-SNE по сырым данным (784)
# ------------------------------
# по 30 изображений каждого класса
indices_per_class = {d: [] for d in range(10)}
for i, label in enumerate(y_test):
    if len(indices_per_class[label]) < 30:
        indices_per_class[label].append(i)
    if all(len(v) == 30 for v in indices_per_class.values()):
        break

selected_indices = np.concatenate([indices_per_class[d] for d in range(10)])  # 300 индексов
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


# ------------------------------
# 7. t-SNE по логитам модели (10)
# ------------------------------
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

print("\nГотово: все пункты задания 0–7 выполнены.")
