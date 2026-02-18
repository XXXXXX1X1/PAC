import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import precision_score, recall_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. Загрузка MNIST данных
mnist = fetch_openml("mnist_784", version=1)
X, y = mnist.data, mnist.target.astype(int)
X = X / 255.0  # нормализация

# Разделение на train/test и преобразование в массивы NumPy
X_train, y_train = X[:60000].values, y[:60000].values
X_test, y_test = X[60000:].values, y[60000:].values

# 2. Рассчитать average_digit для каждой цифры 0-9
avg_digits = []
for digit in range(10):
    mask = y_train == digit
    avg_digit = X_train[mask].mean(axis=0)
    avg_digits.append(avg_digit)


# 3. Классификатор для одной цифры
class SimpleClassifier:
    def __init__(self, avg_digit, bias=0.0):
        self.weights = avg_digit
        self.bias = bias

    def predict(self, x):
        # Простое сравнение: косинусная близость со средним
        similarity = np.dot(x, self.weights) + self.bias
        return 1 if similarity > 0.5 else 0

    def get_similarity(self, x):
        """Возвращает значение сходства (логит)"""
        return np.dot(x, self.weights) + self.bias


# 4. Создать 10 классификаторов
classifiers = [SimpleClassifier(avg_digits[i], bias=-0.1) for i in range(10)]

# 4.1. Рассчитать точность каждого классификатора
print("\nТочность каждого классификатора:")
for i in range(10):
    correct = 0
    total = 0
    # Используем первые 1000 тестовых примеров для скорости
    for j in range(min(1000, len(X_test))):
        x = X_test[j]
        true_label = 1 if y_test[j] == i else 0
        pred_label = classifiers[i].predict(x)
        if pred_label == true_label:
            correct += 1
        total += 1
    accuracy = correct / total
    print(f"Классификатор {i}: {accuracy:.4f}")


# 5. Объединить в одну модель
class MultiDigitModel:
    def __init__ (self, classifiers):
        self.classifiers = classifiers

    def predict_vector(self, x):
        """Возвращает вектор из 10 нулей/единиц"""
        return np.array([clf.predict(x) for clf in self.classifiers])

    def predict_similarity_vector(self, x):
        """Возвращает вектор сходств (логиты)"""
        return np.array([clf.get_similarity(x) for clf in self.classifiers])

    def predict_digit(self, x):
        """Предсказывает одну цифру (0-9)"""
        vector = self.predict_vector(x)
        # Если несколько классификаторов сработали, берём с максимальной уверенностью
        scores = [np.dot(x, clf.weights) for clf in self.classifiers]
        return np.argmax(scores)


# Создаём модель
model = MultiDigitModel(classifiers)

# 6. Тестирование на 1000 примеров (для скорости)
n_test = 1000
y_pred_vectors = []
y_true_vectors = []
y_pred_digits = []

for i in range(n_test):
    x = X_test[i]
    true_digit = y_test[i]

    # Вектор предсказаний
    pred_vector = model.predict_vector(x)
    y_pred_vectors.append(pred_vector)

    # Вектор истинных значений
    true_vector = np.zeros(10)
    true_vector[true_digit] = 1
    y_true_vectors.append(true_vector)

    # Предсказанная цифра
    pred_digit = model.predict_digit(x)
    y_pred_digits.append(pred_digit)

y_pred_vectors = np.array(y_pred_vectors)
y_true_vectors = np.array(y_true_vectors)

# 7. Рассчитать precision и recall
precision = precision_score(
    y_true_vectors, y_pred_vectors, average="macro", zero_division=0
)
recall = recall_score(y_true_vectors, y_pred_vectors, average="macro", zero_division=0)

print(f"\nРезультаты на {n_test} тестовых примерах:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Точность классификации цифр
accuracy = np.mean(np.array(y_pred_digits) == y_test[:n_test])
print(f"Accuracy (цифры): {accuracy:.4f}")

# 8. Визуализация t-SNE (необработанные данные)
# Берём по 30 изображений каждого класса
n_per_class = 30
X_sample = []
y_sample = []

for digit in range(10):
    indices = np.where(y_test == digit)[0][:n_per_class]
    X_sample.extend(X_test[indices])
    y_sample.extend(y_test[indices])

X_sample = np.array(X_sample)
y_sample = np.array(y_sample)

# Применяем t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
X_tsne = tsne.fit_transform(X_sample)

# Рисуем график
plt.figure(figsize=(10, 8))
for digit in range(10):
    mask = y_sample == digit
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=f"Digit {digit}", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("tsne_raw.png")

# 9. Визуализация t-SNE (вектора модели - логиты)
# Получаем вектора логитов (сходства), а не бинарных предсказаний
similarity_vectors = np.array([model.predict_similarity_vector(x) for x in X_sample])

# t-SNE для векторов логитов
similarity_tsne = TSNE(n_components=2, random_state=42, perplexity=20).fit_transform(
    similarity_vectors
)

# Рисуем график
plt.figure(figsize=(10, 8))
for digit in range(10):
    mask = y_sample == digit
    plt.scatter(
        similarity_tsne[mask, 0],
        similarity_tsne[mask, 1],
        label=f"Digit {digit}",
        alpha=0.7,
    )
plt.title("t-SNE: Логиты модели (10-мерные вектора сходства)")
plt.legend()
plt.tight_layout()
plt.savefig("tsne_model.png")

plt.show()
