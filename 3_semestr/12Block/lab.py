import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def dsigmoid_from_y(y):
    # y = sigmoid(z)  =>  dy/dz = y*(1-y)
    return y * (1.0 - y)


class Neuron:
    def __init__(self, input_size, lr=0.1):
        # +1 под bias
        self._weights = np.random.randn(input_size + 1) * 0.5
        self.lr = lr

        self.last_input = None   # x с bias
        self.last_output = None  # y

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        x_ext = np.append(x, 1.0)
        z = np.dot(self._weights, x_ext)
        y = sigmoid(z)

        self.last_input = x_ext
        self.last_output = y
        return y

    def backward(self, x, grad_output):

        x_ext = self.last_input
        y = self.last_output

        dy_dz = dsigmoid_from_y(y)
        dL_dz = grad_output * dy_dz

        # dL/dw = dL/dz * x
        grad_w = dL_dz * x_ext

        # градиент по входу считается по старым весам
        w_old = self._weights.copy()

        # шаг градиентного спуска
        self._weights -= self.lr * grad_w

        # dL/dx = dL/dz * w
        grad_input = dL_dz * w_old[:-1]
        return grad_input


class Model:
    def __init__(self, lr=0.1, seed=1):
        np.random.seed(seed)
        self.h1 = Neuron(2, lr=lr)
        self.h2 = Neuron(2, lr=lr)
        self.out = Neuron(2, lr=lr)

    def forward(self, x):
        h1 = self.h1.forward(x)
        h2 = self.h2.forward(x)
        y = self.out.forward([h1, h2])
        return y

    def backward(self, x, err):

        hidden = np.array([self.h1.last_output, self.h2.last_output], dtype=float)

        grad_hidden = self.out.backward(hidden, err)   # вернёт [dL/dh1, dL/dh2]
        self.h1.backward(x, grad_hidden[0])
        self.h2.backward(x, grad_hidden[1])


def mse(y, t):
    return 0.5 * (y - t) ** 2

def d_mse_dy(y, t):
    return (y - t)


if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    Y = np.array([0,1,1,0], dtype=float)

    model = Model(lr=0.1, seed=42)

    epochs = 100000
    for epoch in range(epochs):
        total_loss = 0.0

        for x, t in zip(X, Y):
            y_pred = model.forward(x)
            total_loss += mse(y_pred, t)

            err = d_mse_dy(y_pred, t)      # это dL/dy
            model.backward(x, err)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch:6d} | total_loss = {total_loss:.6f}")

    print("\nPredictions:")
    for x, t in zip(X, Y):
        y_pred = model.forward(x)
        print(x, "->", float(y_pred), "target", t)
