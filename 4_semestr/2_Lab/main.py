import os

# ==========================
# 0) ГЛУШИМ ЛИШНИЕ ЛОГИ TF
# (важно: ДО import tensorflow)
# ==========================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # убирает oneDNN-спам
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"  # выключаем авто-XLA

import random
import numpy as np
import imageio.v2 as imageio

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Lambda, Dense, Conv2D, MaxPooling2D, Flatten,
    BatchNormalization, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

# -----------------------
# Конфиг
# -----------------------
DATA_DIR = "/home/alex/PycharmProjects/PAC/4_semestr/2_Lab/orl_faces"
SEED = 42

N_TRAIN_PERSONS = 20
N_VAL_PERSONS = 10
N_TEST_PERSONS = 10

TOTAL_POS_PER_PERSON = 250

BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
MARGIN = 1.0


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def print_device_info():
    gpus = tf.config.list_physical_devices("GPU")
    print("device: cuda" if gpus else "device: cpu")


def load_orl_images(data_dir: str) -> np.ndarray:
    imgs = []
    for person in range(1, 41):
        row = []
        for idx in range(1, 11):
            path = os.path.join(data_dir, f"s{person}", f"{idx}.pgm")
            img = imageio.imread(path)
            row.append(img)
        imgs.append(row)
    return np.array(imgs, dtype=np.uint8)


def make_pairs_from_persons(imgs: np.ndarray, person_ids, total_pos_per_person: int, seed: int):
    rng = np.random.default_rng(seed)

    person_ids = np.array(person_ids, dtype=np.int32)
    n_persons = len(person_ids)
    n_imgs = imgs.shape[1]
    H, W = imgs.shape[2], imgs.shape[3]

    total_pos = n_persons * total_pos_per_person
    total_neg = total_pos
    total = total_pos + total_neg

    X = np.empty((total, 2, H, W, 1), dtype=np.uint8)
    Y = np.empty((total, 1), dtype=np.float32)

    k = 0
    # POS
    for pid in person_ids:
        for _ in range(total_pos_per_person):
            a, b = rng.choice(n_imgs, size=2, replace=False)
            X[k, 0, :, :, 0] = imgs[pid, a]
            X[k, 1, :, :, 0] = imgs[pid, b]
            Y[k, 0] = 1.0
            k += 1

    # NEG
    for _ in range(total_neg):
        p1, p2 = rng.choice(person_ids, size=2, replace=False)
        a = rng.integers(0, n_imgs)
        b = rng.integers(0, n_imgs)
        X[k, 0, :, :, 0] = imgs[p1, a]
        X[k, 1, :, :, 0] = imgs[p2, b]
        Y[k, 0] = 0.0
        k += 1

    idx = rng.permutation(total)
    X = X[idx].astype(np.float32) / 255.0
    Y = Y[idx]
    return X, Y


def build_base_network(input_shape):
    inp = Input(shape=input_shape)

    x = Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = MaxPooling2D(2)(x)

    x = Conv2D(64, 3, padding="same", activation="relu")(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(128, 3, padding="same", activation="relu")(x)
    x = MaxPooling2D(2)(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Dense(64)(x)  # без relu
    x = Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    return Model(inp, x, name="base_network")


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
    d = distances.ravel()
    candidates = np.quantile(d, np.linspace(0.01, 0.99, 99))
    best_t, best_acc = float(candidates[0]), -1.0
    for t in candidates:
        acc = compute_accuracy(distances, labels, threshold=float(t))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


class PrintEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs: int):
        super().__init__()
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = float(logs.get("loss", 0.0))
        val_loss = float(logs.get("val_loss", 0.0))
        ep = epoch + 1
        print(f"ep {ep:02d}/{self.epochs} | loss={loss:.4f} | val_loss={val_loss:.4f}")


def main():
    seed_all(SEED)
    print_device_info()

    imgs = load_orl_images(DATA_DIR)

    rng = np.random.default_rng(SEED)
    persons = np.arange(40, dtype=np.int32)
    rng.shuffle(persons)

    train_persons = persons[:N_TRAIN_PERSONS]
    val_persons = persons[N_TRAIN_PERSONS:N_TRAIN_PERSONS + N_VAL_PERSONS]
    test_persons = persons[N_TRAIN_PERSONS + N_VAL_PERSONS:
                           N_TRAIN_PERSONS + N_VAL_PERSONS + N_TEST_PERSONS]

    print("train persons:", train_persons)
    print("val persons  :", val_persons)
    print("test persons :", test_persons)

    X_train, Y_train = make_pairs_from_persons(imgs, train_persons, TOTAL_POS_PER_PERSON, seed=SEED)
    X_val, Y_val     = make_pairs_from_persons(imgs, val_persons,   TOTAL_POS_PER_PERSON, seed=SEED + 1)
    X_test, Y_test   = make_pairs_from_persons(imgs, test_persons,  TOTAL_POS_PER_PERSON, seed=SEED + 2)

    input_dim = X_train.shape[2:]

    img_a = Input(shape=input_dim)
    img_b = Input(shape=input_dim)

    base_network = build_base_network(input_dim)
    feat_a = base_network(img_a)
    feat_b = base_network(img_b)

    dist = Lambda(euclidean_distance)([feat_a, feat_b])

    model = Model([img_a, img_b], dist)
    model.compile(
        optimizer=RMSprop(learning_rate=LR),
        loss=contrastive_loss(margin=MARGIN)
    )

    Xtr0, Xtr1 = X_train[:, 0], X_train[:, 1]
    Xv0, Xv1   = X_val[:, 0],   X_val[:, 1]
    Xte0, Xte1 = X_test[:, 0],  X_test[:, 1]

    model.fit(
        [Xtr0, Xtr1], Y_train,
        validation_data=([Xv0, Xv1], Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=0,
        callbacks=[PrintEpochCallback(EPOCHS)]
    )

    pred_val = model.predict([Xv0, Xv1], verbose=0)
    best_t, best_acc_val = find_best_threshold(pred_val, Y_val)

    pred_test = model.predict([Xte0, Xte1], verbose=0)
    acc_best = compute_accuracy(pred_test, Y_test, threshold=best_t)

    print(f"\nVAL best_t={best_t:.6f} | VAL acc={best_acc_val:.4f}")
    print(f"TEST acc@best_t={acc_best:.4f}")


if __name__ == "__main__":
    main()