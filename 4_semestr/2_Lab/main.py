import os
import re
import numpy as np

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop


# --- важно для TF на CPU ---
tf.keras.backend.set_image_data_format("channels_last")


# ---------- PGM reader (P5) ----------
def read_pgm(path: str) -> np.ndarray:
    """
    Reads binary PGM (P5) and returns uint8 array (H, W).
    ORL/ATT faces dataset compatible.
    """
    with open(path, "rb") as f:
        buf = f.read()

    m = re.search(
        br"^P5\s(?:\s*#.*[\r\n])*"
        br"(\d+)\s(?:\s*#.*[\r\n])*"
        br"(\d+)\s(?:\s*#.*[\r\n])*"
        br"(\d+)\s",
        buf
    )
    if m is None:
        raise ValueError(f"Not a valid binary PGM (P5): {path}")

    w = int(m.group(1))
    h = int(m.group(2))
    maxval = int(m.group(3))
    offset = m.end()  # конец заголовка

    if maxval < 256:
        img = np.frombuffer(buf, dtype=np.uint8, count=w * h, offset=offset)
    else:
        img = np.frombuffer(buf, dtype=">u2", count=w * h, offset=offset)

    return img.reshape((h, w))


# ---------- Pair generator (channels_last) ----------
def get_pairs_orl(data_dir: str, downsample: int = 2, total_pos_per_person: int = 250, seed: int = 42):
    """
    Output:
      X: (N, 2, H, W, 1) float32 in [0,1]
      Y: (N, 1) float32 labels (1 same, 0 different)
    """
    rng = np.random.default_rng(seed)

    probe = read_pgm(os.path.join(data_dir, "s1", "1.pgm"))[::downsample, ::downsample]
    H, W = probe.shape

    n_persons = 40
    n_imgs = 10

    total_pos = n_persons * total_pos_per_person
    total_neg = total_pos  # баланс

    X_pos = np.zeros((total_pos, 2, H, W, 1), dtype=np.float32)
    y_pos = np.ones((total_pos, 1), dtype=np.float32)

    # positive pairs (same person)
    k = 0
    for person in range(1, n_persons + 1):
        for _ in range(total_pos_per_person):
            a = b = 0
            while a == b:
                a = rng.integers(1, n_imgs + 1)
                b = rng.integers(1, n_imgs + 1)

            img1 = read_pgm(os.path.join(data_dir, f"s{person}", f"{a}.pgm"))[::downsample, ::downsample]
            img2 = read_pgm(os.path.join(data_dir, f"s{person}", f"{b}.pgm"))[::downsample, ::downsample]

            X_pos[k, 0, :, :, 0] = img1
            X_pos[k, 1, :, :, 0] = img2
            k += 1

    X_neg = np.zeros((total_neg, 2, H, W, 1), dtype=np.float32)
    y_neg = np.zeros((total_neg, 1), dtype=np.float32)

    # negative pairs (different persons)
    k = 0
    for _ in range(total_neg):
        p1 = p2 = 1
        while p1 == p2:
            p1 = rng.integers(1, n_persons + 1)
            p2 = rng.integers(1, n_persons + 1)

        a = rng.integers(1, n_imgs + 1)
        b = rng.integers(1, n_imgs + 1)

        img1 = read_pgm(os.path.join(data_dir, f"s{p1}", f"{a}.pgm"))[::downsample, ::downsample]
        img2 = read_pgm(os.path.join(data_dir, f"s{p2}", f"{b}.pgm"))[::downsample, ::downsample]

        X_neg[k, 0, :, :, 0] = img1
        X_neg[k, 1, :, :, 0] = img2
        k += 1

    X = np.concatenate([X_pos, X_neg], axis=0) / 255.0
    Y = np.concatenate([y_pos, y_neg], axis=0)

    # перемешаем
    idx = rng.permutation(X.shape[0])
    return X[idx], Y[idx]


# ---------- Base CNN (shared) ----------
def build_base_network(input_shape):
    # input_shape: (H, W, 1)  -> channels_last
    return Sequential([
        Input(shape=input_shape),

        Conv2D(6, (3, 3), activation="relu", padding="valid"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(12, (3, 3), activation="relu", padding="valid"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.1),
        Dense(50, activation="relu"),  # embedding
    ])


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        return K.mean(
            y_true * K.square(y_pred) +
            (1.0 - y_true) * K.square(K.maximum(margin - y_pred, 0.0))
        )
    return loss


def compute_accuracy(distances, labels, threshold=0.5):
    preds_same = (distances.ravel() < threshold).astype(np.float32)
    return float(np.mean(preds_same == labels.ravel()))


def find_best_threshold(distances, labels):
    # грубо, но быстро: перебор порогов по квантилям расстояний
    d = distances.ravel()
    y = labels.ravel()
    candidates = np.quantile(d, np.linspace(0.01, 0.99, 99))
    best_t, best_acc = candidates[0], -1.0
    for t in candidates:
        acc = compute_accuracy(distances, labels, threshold=float(t))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def main():
    # Поменяй на свой путь
    data_dir = "/home/alex/PycharmProjects/PAC/4_semestr/2_Lab/orl_faces"

    X, Y = get_pairs_orl(data_dir, downsample=2, total_pos_per_person=250, seed=42)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=0.2,
        random_state=42,
        stratify=Y.ravel().astype(int)
    )

    input_dim = x_train.shape[2:]  # (H, W, 1)

    img_a = Input(shape=input_dim)
    img_b = Input(shape=input_dim)

    base_network = build_base_network(input_dim)
    feat_a = base_network(img_a)
    feat_b = base_network(img_b)

    dist = Lambda(euclidean_distance)([feat_a, feat_b])

    model = Model([img_a, img_b], dist)
    model.compile(optimizer=RMSprop(learning_rate=1e-3), loss=contrastive_loss(margin=1.0))

    model.fit(
        [x_train[:, 0], x_train[:, 1]],
        y_train,
        validation_split=0.25,
        batch_size=128,
        epochs=13,
        verbose=2
    )

    pred = model.predict([x_test[:, 0], x_test[:, 1]], verbose=0)

    # точность с порогом 0.5 + лучший порог (под тест, чисто для оценки)
    acc_05 = compute_accuracy(pred, y_test, threshold=0.5)
    best_t, best_acc = find_best_threshold(pred, y_test)

    print("Test accuracy (threshold=0.5):", acc_05)
    print("Best threshold:", best_t, "Best test accuracy:", best_acc)


if __name__ == "__main__":
    main()