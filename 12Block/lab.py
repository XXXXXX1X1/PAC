import numpy as np


# ======== ВСПОМОГАТЕЛЬНОЕ ========

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ======== КЛАСС НЕЙРОНА ========

class Neuron:
    def __init__(self, input_size: int, lr: float = 0.1):
        """
        input_size  – количество входов нейрона (без учёта bias)
        lr          – скорость обучения
        """
        # +1 вес под bias
        self._weights = np.random.randn(input_size + 1) * 0.1
        self.lr = lr

        # Состояние нейрона для backprop
        self.last_input = None   # вектор x (с добавленным bias)
        self.last_output = None  # y = sigmoid(z)

    def forward(self, x):
        """
        x: одномерный вектор входов (без bias).
        Возвращает скалярный выход нейрона.
        """
        x = np.asarray(x, dtype=float)
        # добавляем bias = 1.0
        x_ext = np.append(x, 1.0)

        self.last_input = x_ext
        z = np.dot(self._weights, x_ext)
        self.last_output = sigmoid(z)
        return self.last_output

    def backward(self, x, grad_output):
        """
        x           – вход (тот же, что в forward), но мы в реальности
                      используем сохранённый last_input.
        grad_output – dL/dy, градиент функции потерь по выходу нейрона.

        Обновляет веса нейрона.
        Возвращает градиент по входу: dL/dx (без bias-компоненты).
        """
        # Берём сохранённый вход (уже с bias)
        x_ext = self.last_input

        # y = sigmoid(z),  dy/dz = y * (1 - y)
        dy_dz = self.last_output * (1.0 - self.last_output)

        # dL/dz = dL/dy * dy/dz
        dz = grad_output * dy_dz

        # Градиент по весам: dL/dw_i = dL/dz * x_i
        grad_w = dz * x_ext

        # Обновляем веса
        self._weights -= self.lr * grad_w

        # Градиент по входу (без bias): dL/dx_i = dL/dz * w_i
        grad_input = self._weights[:-1] * dz
        return grad_input


# ======== МОДЕЛЬ: 2 нейрона в скрытом слое + 1 выходной ========

class Model:
    def __init__(self, lr: float = 0.1, seed: int = 0):
        np.random.seed(seed)
        # Скрытый слой: 2 нейрона по 2 входа
        self.h1 = Neuron(input_size=2, lr=lr)
        self.h2 = Neuron(input_size=2, lr=lr)
        # Выходной слой: 1 нейрон, 2 входа (выходы h1, h2)
        self.out = Neuron(input_size=2, lr=lr)

    def forward(self, x):
        """
        x – вектор размера 2 (вход XOR).
        Возвращает скаляр – предсказание модели.
        """
        x = np.asarray(x, dtype=float)

        # Прямой проход по скрытому слою
        h1_out = self.h1.forward(x)
        h2_out = self.h2.forward(x)
        hidden = np.array([h1_out, h2_out])

        # Прямой проход по выходному нейрону
        y = self.out.forward(hidden)
        return y

    def backward(self, x, grad_loss_out):
        """
        x             – исходный вход XOR (2 числа)
        grad_loss_out – dL/dy_out, градиент потерь по выходу сети.
        """
        x = np.asarray(x, dtype=float)

        # Вектор входа выходного нейрона — это выходы скрытого слоя
        hidden = np.array([self.h1.last_output, self.h2.last_output])

        # Backward для выходного нейрона: получаем градиент по hidden
        grad_hidden = self.out.backward(hidden, grad_loss_out)
        # grad_hidden[0] = dL/dh1_out, grad_hidden[1] = dL/dh2_out

        # Гоним градиент дальше в скрытый слой
        self.h1.backward(x, grad_hidden[0])
        self.h2.backward(x, grad_hidden[1])


# ======== ОБУЧЕНИЕ НА XOR ========

if __name__ == "__main__":
    # Обучающая выборка для XOR
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)

    y_true = np.array([0, 1, 1, 0], dtype=float)

    # Модель
    model = Model(lr=0.1, seed=0)

    # MSE = 0.5 * (y - t)^2
    def mse(y, t):
        return 0.5 * (y - t) ** 2

    # Градиент MSE по выходу: dL/dy = y - t
    def loss_grad(y, t):
        return y - t

    epochs = 40000

    for epoch in range(epochs):
        total_loss = 0.0
        for x, t in zip(X, y_true):
            # Прямой проход
            y_pred = model.forward(x)
            # Скалярная ошибка для логов
            total_loss += mse(y_pred, t)
            # Градиент по выходу
            err = loss_grad(y_pred, t)
            # Обратный проход
            model.backward(x, err)

        if epoch % 5000 == 0:
            print(f"Epoch {epoch:5d}, loss = {float(total_loss):.6f}")

    # ===== Проверка =====
    print("\nИтоговые предсказания XOR после обучения:\n")
    for x, t in zip(X, y_true):
        y_pred = model.forward(x)
        print(f"input {x}  ->  pred = {y_pred:.4f}, target = {t}")
