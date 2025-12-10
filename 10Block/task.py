
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score
from sklearn.manifold import TSNE
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torchvision


plt.rcParams['figure.figsize'] = (6, 6)

# ------------------------------
# 0. Загрузка MNIST через torchvision
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

print("train size:", len(train_dataset))
print("test size :", len(test_dataset))


# ------------------------------
# 1. Преобразование датасета в массивы (N,784) и метки (N,)
# ------------------------------
def dataset_to_arrays(dataset):
    """
    dataset: torchvision MNIST
    возвращает:
        X — np.array (N, 784)
        y — np.array (N,)
    """
    X = []
    y = []
    for img, label in dataset:
        # img: tensor 1x28x28
        arr = img[0].numpy().reshape(-1)   # (784,)
        X.append(arr)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


X_train, y_train = dataset_to_arrays(train_dataset)
X_test, y_test   = dataset_to_arrays(test_dataset)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)


# ------------------------------
# 2. average_digit для каждой цифры 0–9
# ------------------------------
def average_digit(X, y, digit):
    """
    Средний вектор (784,) по всем картинкам класса digit.
    """
    mask = (y == digit)
    return X[mask].mean(axis=0)


avg_digits = np.stack([average_digit(X_train, y_train, d) for d in range(10)])  # (10, 784)

# Визуализация средних цифр (можно закомментировать, если не нужно)
fig, axes = plt.subplots(2, 5, figsize=(8, 4))
for d in range(10):
    ax = axes[d // 5, d % 5]
    ax.imshow(avg_digits[d].reshape(28, 28), cmap="gray")
    ax.set_title(str(d))
    ax.axis("off")
plt.tight_layout()
plt.show()


# ------------------------------
# 3. Десять бинарных классификаторов (one-vs-all) и их bias
# ------------------------------
biases = np.zeros(10, dtype=np.float32)

for d in range(10):
    w = avg_digits[d]                         # (784,)
    scores = X_train @ w                      # (N,)

    pos_scores = scores[y_train == d]
    neg_scores = scores[y_train != d]

    # порог — середина между средними значениями для своего и чужих классов
    thr = 0.5 * (pos_scores.mean() + neg_scores.mean())
    b = -thr
    biases[d] = b


def predict_binary_digit(x_vec, d):
    """
    x_vec: (784,)
    d: цифра 0..9
    возвращает 0 или 1
    """
    w = avg_digits[d]
    b = biases[d]
    score = np.dot(w, x_vec) + b
    return 1 if score >= 0 else 0


digit_accuracies = []

print("\nТочность бинарных классификаторов (one-vs-all) на тесте:")
for d in range(10):
    correct = 0
    total = 0
    for i in range(len(X_test)):
        x = X_test[i]
        y_true = 1 if y_test[i] == d else 0
        y_pred = predict_binary_digit(x, d)
        if y_true == y_pred:
            correct += 1
        total += 1
    acc = correct / total
    digit_accuracies.append(acc)
    print(f"Классификатор цифры {d}: accuracy = {acc:.4f}")


# ------------------------------
# 4. Объединённая модель: логиты (10) и предсказание класса
# ------------------------------
def model_logits(x_vec):
    """
    x_vec: (784,)
    возвращает np.array shape (10,) — логиты по всем классам
    """
    # avg_digits: (10,784), biases: (10,)
    return avg_digits @ x_vec + biases


def model_predict_class(x_vec):
    """
    возвращает предсказанную цифру 0..9
    """
    scores = model_logits(x_vec)
    return int(np.argmax(scores))


# Проверка на одном примере (можно убрать)
idx = 0
plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()
print("Пример:")
print("  Истинный label:", y_test[idx])
print("  Предсказание  :", model_predict_class(X_test[idx]))
print("  Логиты        :", model_logits(X_test[idx]))


# ------------------------------
# 5. Accuracy, precision, recall для многоклассовой модели
# ------------------------------
y_pred_test = np.array([model_predict_class(x) for x in X_test])

overall_acc = np.mean(y_pred_test == y_test)
print(f"\nОбщая accuracy (10 классов) на тесте: {overall_acc:.4f}")

precision_micro = precision_score(y_test, y_pred_test, average='micro')
recall_micro    = recall_score(y_test, y_pred_test, average='micro')

precision_macro = precision_score(y_test, y_pred_test, average='macro')
recall_macro    = recall_score(y_test, y_pred_test, average='macro')

print(f"Precision (micro): {precision_micro:.4f}")
print(f"Recall    (micro): {recall_micro:.4f}")
print(f"Precision (macro): {precision_macro:.4f}")
print(f"Recall    (macro): {recall_macro:.4f}")


# ------------------------------
# 6. t-SNE по сырым данным (784)
# ------------------------------
# Берём по 30 изображений каждого класса из теста
indices_per_class = {d: [] for d in range(10)}

for i, label in enumerate(y_test):
    if len(indices_per_class[label]) < 30:
        indices_per_class[label].append(i)
    if all(len(v) == 30 for v in indices_per_class.values()):
        break

selected_indices = []
for d in range(10):
    selected_indices.extend(indices_per_class[d])

selected_indices = np.array(selected_indices)  # 300 индексов

X_raw = X_test[selected_indices]   # (300, 784)
y_raw = y_test[selected_indices]   # (300,)

print("\nЗапускаем t-SNE по сырым данным (784)...")
tsne_raw = TSNE(
    n_components=2,
    init='random',
    learning_rate='auto',
    random_state=42,
)

X_raw_2d = tsne_raw.fit_transform(X_raw)  # (300,2)

plt.figure(figsize=(7, 7))
for d in range(10):
    mask = (y_raw == d)
    plt.scatter(X_raw_2d[mask, 0], X_raw_2d[mask, 1], s=10, label=str(d))
plt.legend()
plt.title("t-SNE по сырым пикселям (784)")
plt.show()


# ------------------------------
# 7. t-SNE по логитам модели (10)
# ------------------------------
print("Запускаем t-SNE по логитам модели (10)...")
logits = []
for i in selected_indices:
    x = X_test[i]
    logits.append(model_logits(x))
logits = np.array(logits)   # (300, 10)

tsne_logits = TSNE(
    n_components=2,
    init='random',
    learning_rate='auto',
    random_state=42,
)

logits_2d = tsne_logits.fit_transform(logits)

plt.figure(figsize=(7, 7))
for d in range(10):
    mask = (y_raw == d)
    plt.scatter(logits_2d[mask, 0], logits_2d[mask, 1], s=10, label=str(d))
plt.legend()
plt.title("t-SNE по выходам модели (логиты, 10)")
plt.show()
