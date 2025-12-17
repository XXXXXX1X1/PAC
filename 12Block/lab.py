import numpy as np

# =========================
# 1) Вспомогательные вещи
# =========================

def sigmoid(z: float) -> float:
    """Сигмоида: переводит число в диапазон (0..1)."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative_from_output(y: float) -> float:
    """
    Производная сигмоиды, если мы уже знаем y = sigmoid(z).
    d/dz sigmoid(z) = y * (1 - y)
    """
    return y * (1.0 - y)


# =========================
# 2) Один нейрон
# =========================

class Neuron:
    def __init__(self, input_size: int, lr: float = 0.1):
        """
        input_size: сколько "обычных" входов (без bias).
        lr: скорость обучения (шаг градиентного спуска).
        """
        self.lr = lr

        # +1 вес под bias, потому что bias мы будем добавлять как дополнительный вход "1.0"
        self.w = np.random.randn(input_size + 1) * 0.1

        # Сохраняем данные последнего forward — это нужно для backward
        self.last_x_ext = None   # вход с bias: [x1, x2, 1.0]
        self.last_y = None       # выход y
        self.last_z = None       # линейная сумма z

    def forward(self, x):
        """
        Прямой проход.
        x: вход без bias, например [0, 1] или [h1_out, h2_out]
        """
        x = np.asarray(x, dtype=float)

        # Добавляем bias как дополнительный вход = 1.0
        x_ext = np.append(x, 1.0)

        # Линейная часть: z = w·x
        z = float(np.dot(self.w, x_ext))

        # Нелинейность
        y = sigmoid(z)

        # Сохраняем для backward
        self.last_x_ext = x_ext
        self.last_z = z
        self.last_y = y

        return y

    def backward(self, grad_L_by_y):
        """
        Обратный проход для одного нейрона.

        grad_L_by_y = dL/dy (градиент потерь по выходу нейрона).
        Возвращаем dL/dx (по входам НЕ включая bias).

        Пара слов:
        y = sigmoid(z)
        z = sum(w_i * x_i)
        """

        # 1) Берём сохранённые значения с forward
        x_ext = self.last_x_ext
        y = self.last_y

        # 2) Считаем dL/dz через цепное правило:
        #    dL/dz = dL/dy * dy/dz
        dy_dz = sigmoid_derivative_from_output(y)
        grad_L_by_z = grad_L_by_y * dy_dz

        # 3) Градиент по весам:
        #    dL/dw_i = dL/dz * x_i
        grad_w = grad_L_by_z * x_ext

        # ⚠️ Важный момент:
        # Градиент по входу должен считаться по "старым" весам (до обновления).
        w_old = self.w.copy()

        # 4) Обновляем веса градиентным спуском:
        #    w := w - lr * dL/dw
        self.w -= self.lr * grad_w

        # 5) Градиент по входу (без bias):
        #    dL/dx_i = dL/dz * w_i
        #    bias-вес не возвращаем, поэтому w_old[:-1]
        grad_x = grad_L_by_z * w_old[:-1]

        return grad_x


# =========================
# 3) Модель 2-2-1 для XOR
# =========================

class XORModel:
    def __init__(self, lr: float = 0.1, seed: int = 42):
        np.random.seed(seed)

        # Скрытый слой: два нейрона, каждый принимает 2 входа (x1, x2)
        self.h1 = Neuron(input_size=2, lr=lr)
        self.h2 = Neuron(input_size=2, lr=lr)

        # Выходной нейрон: принимает 2 входа (h1_out, h2_out)
        self.out = Neuron(input_size=2, lr=lr)

        # Для удобства сохраним последний hidden
        self.last_hidden = None

    def forward(self, x):
        """
        x: [x1, x2]
        """
        x = np.asarray(x, dtype=float)

        h1_out = self.h1.forward(x)
        h2_out = self.h2.forward(x)

        hidden = np.array([h1_out, h2_out], dtype=float)
        self.last_hidden = hidden

        y_pred = self.out.forward(hidden)
        return y_pred

    def backward(self, grad_L_by_y_pred):
        """
        grad_L_by_y_pred = dL/dy_pred
        Сначала обучаем выходной нейрон, потом скрытые.
        """

        # 1) Прогоняем backward через выходной нейрон
        #    Получаем градиент по hidden: [dL/dh1_out, dL/dh2_out]
        grad_hidden = self.out.backward(grad_L_by_y_pred)

        # 2) Прогоняем градиенты дальше в скрытый слой
        #    Тут каждый скрытый нейрон получает "свой" градиент
        _ = self.h1.backward(grad_hidden[0])
        _ = self.h2.backward(grad_hidden[1])


# =========================
# 4) Обучение XOR
# =========================

def mse_loss(y_pred: float, y_true: float) -> float:
    """MSE = 0.5 * (y - t)^2 (так удобнее для производной)."""
    return 0.5 * (y_pred - y_true) ** 2

def grad_mse_by_pred(y_pred: float, y_true: float) -> float:
    """d/dy [0.5*(y-t)^2] = (y - t)"""
    return (y_pred - y_true)


if __name__ == "__main__":
    # XOR данные
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ], dtype=float)

    y = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)

    model = XORModel(lr=0.1, seed=42)

    epochs = 1000000

    # Если хочешь “наглядно”, можно включить подробные логи на первых эпохах
    VERBOSE = False

    for epoch in range(epochs):
        total_loss = 0.0

        for x_i, t_i in zip(X, y):
            # 1) forward
            y_pred = model.forward(x_i)

            # 2) loss
            total_loss += mse_loss(y_pred, t_i)

            # 3) градиент loss по выходу сети
            grad = grad_mse_by_pred(y_pred, t_i)

            # 4) backward (обновит веса)
            model.backward(grad)

            if VERBOSE and epoch < 2:
                print(f"x={x_i}, t={t_i}, y_pred={y_pred:.4f}, loss={mse_loss(y_pred,t_i):.6f}")

        if epoch % 2000 == 0:
            print(f"Epoch {epoch:5d} | total_loss = {total_loss:.6f}")

    # Проверка
    print("\nИтоговые предсказания XOR:\n")
    for x_i, t_i in zip(X, y):
        y_pred = model.forward(x_i)
        print(f"input {x_i} -> pred={y_pred:.4f}, target={t_i}")
