import os

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

import matplotlib.pyplot as plt


DATA_DIR = "/Users/xxx/Desktop/Учеба/Python/Pac/4_semestr/2_Lab/orl_faces"  # путь к ORL (s1..s40)
SEED = 445  # фиксируем случайность для воспроизводимости

# split по людям (без утечки): train/val/test содержат разных людей
N_TRAIN_PERSONS = 25
N_VAL_PERSONS = 8
N_TEST_PERSONS = 7

TOTAL_POS_PER_PERSON = 250  # сколько положительных пар на одного человека (NEG делаем столько же)

BATCH_SIZE = 32  # сколько пар за один шаг обучения
EPOCHS = 50      # число эпох
LR = 1e-4       # learning rate оптимизатора
MARGIN = 1.0     # margin для contrastive loss (насколько раздвигать отрицательные пары)

OUT_DIR = "./out_vis"
os.makedirs(OUT_DIR, exist_ok=True)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def print_device_info():
    gpus = tf.config.list_physical_devices("GPU")
    print("device: cuda" if gpus else "device: cpu")


# ЗАГРУЗКА ORL (40 людей × 10 фото)
def load_orl_images(data_dir: str) -> np.ndarray:
    imgs = []
    for person in range(1, 41):  # s1..s40
        row = []
        for idx in range(1, 11):  # 1..10.pgm
            path = os.path.join(data_dir, f"s{person}", f"{idx}.pgm")
            img = imageio.imread(path)
            row.append(img)
        imgs.append(row)
    return np.array(imgs, dtype=np.uint8)  # (40,10,H,W)


# ГЕНЕРАЦИЯ ПАР
def make_pairs_from_persons(imgs: np.ndarray, person_ids, total_pos_per_person: int, seed: int):
    rng = np.random.default_rng(seed)

    person_ids = np.array(person_ids, dtype=np.int32)
    n_persons = len(person_ids)
    n_imgs = imgs.shape[1]
    H, W = imgs.shape[2], imgs.shape[3]

    total_pos = n_persons * total_pos_per_person
    total_neg = total_pos  # баланс: POS = NEG
    total = total_pos + total_neg

    X = np.empty((total, 2, H, W, 1), dtype=np.uint8)  # пары изображений
    Y = np.empty((total, 1), dtype=np.float32)         # метки 1/0 (same/diff)

    k = 0

    # POS (same person)
    for pid in person_ids:
        for _ in range(total_pos_per_person):
            a, b = rng.choice(n_imgs, size=2, replace=False)
            X[k, 0, :, :, 0] = imgs[pid, a]
            X[k, 1, :, :, 0] = imgs[pid, b]
            Y[k, 0] = 1.0
            k += 1

    # NEG (different persons)
    for _ in range(total_neg):
        p1, p2 = rng.choice(person_ids, size=2, replace=False)
        a = rng.integers(0, n_imgs)
        b = rng.integers(0, n_imgs)
        X[k, 0, :, :, 0] = imgs[p1, a]
        X[k, 1, :, :, 0] = imgs[p2, b]
        Y[k, 0] = 0.0
        k += 1

    # shuffle + normalize to [0,1]
    idx = rng.permutation(total)
    X = X[idx].astype(np.float32) / 255.0
    Y = Y[idx]
    return X, Y


# BASE NETWORK: картинка -> embedding (нормированный)
def build_base_network(input_shape):
    inp = Input(shape=input_shape)  # вход: одна картинка (H, W, 1)

    x = Conv2D(32, 3, padding="same", activation="relu")(inp)  # извлекаем простые признаки
    x = MaxPooling2D(2)(x)                                     # уменьшаем размер (быстрее/устойчивее)

    x = Conv2D(64, 3, padding="same", activation="relu")(x)    # извлекаем более сложные признаки
    x = MaxPooling2D(2)(x)                                     # уменьшаем размер

    x = Conv2D(128, 3, padding="same", activation="relu")(x)   # ещё более сложные признаки
    x = MaxPooling2D(2)(x)                                     # уменьшаем размер

    x = Flatten()(x)                                           # превращаем карты признаков в вектор
    x = Dense(128, activation="relu")(x)                       # собираем признаки в компактный вид
    x = BatchNormalization()(x)                                # стабилизируем обучение
    x = Dropout(0.1)(x)                                        # уменьшаем переобучение

    x = Dense(64)(x)                                           # итоговый embedding (вектор признаков)
    x = Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)    # нормируем embedding для корректного сравнения
    return Model(inp, x, name="base_network")                  # одна ветка сиамской сети (веса общие)


# DISTANCE + CONTRASTIVE LOSS
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def contrastive_loss(margin=1.0):
    # POS: тянем distance -> 0
    # NEG: раздвигаем distance >= margin
    def loss(y_true, y_pred):
        return K.mean(
            y_true * K.square(y_pred) +
            (1.0 - y_true) * K.square(K.maximum(margin - y_pred, 0.0))
        )
    return loss


# ACCURACY ПО ПОРОГУ (threshold подбираем по VAL)
def compute_accuracy(distances, labels, threshold=0.5):
    preds_same = (distances.ravel() < threshold).astype(np.float32)  # dist<thr => same
    return float(np.mean(preds_same == labels.ravel()))

def find_best_threshold(distances, labels):
    # ищем порог, который даёт максимум accuracy на VAL
    d = distances.ravel()
    candidates = np.quantile(d, np.linspace(0.01, 0.99, 99))
    best_t, best_acc = float(candidates[0]), -1.0
    for t in candidates:
        acc = compute_accuracy(distances, labels, threshold=float(t))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def save_inference_grid(X_pairs, dists, threshold, save_path, num_pairs=10, seed=42):
    rng = np.random.default_rng(seed)
    d = dists.ravel()
    n = X_pairs.shape[0]
    num_pairs = min(num_pairs, n)

    idx = rng.choice(n, size=num_pairs, replace=False)

    pairs_per_row = 2
    rows = int(np.ceil(num_pairs / pairs_per_row))
    cols = pairs_per_row * 2

    plt.figure(figsize=(10, 2.8 * rows))
    plt.suptitle(f"Threshold = {threshold:.4f} (dist < thr => SAME)", y=0.99)

    for i, j in enumerate(idx):
        r = i // pairs_per_row
        p = i % pairs_per_row
        base = r * cols + p * 2

        a = X_pairs[j, 0, :, :, 0]
        b = X_pairs[j, 1, :, :, 0]
        dist = float(d[j])
        pred = "SAME" if dist < threshold else "DIFF"

        ax1 = plt.subplot(rows, cols, base + 1)
        ax1.imshow(a, cmap="gray", vmin=0.0, vmax=1.0)
        ax1.axis("off")
        ax1.set_title(f"{pred} | Distance: {dist:.2f}")

        ax2 = plt.subplot(rows, cols, base + 2)
        ax2.imshow(b, cmap="gray", vmin=0.0, vmax=1.0)
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    seed_all(SEED)
    print_device_info()

    imgs = load_orl_images(DATA_DIR)  # грузим ORL: (40, 10, H, W)

    # делим людей на train/val/test
    rng = np.random.default_rng(SEED)
    persons = np.arange(40, dtype=np.int32)
    rng.shuffle(persons)

    train_persons = persons[:N_TRAIN_PERSONS]                                      # люди для train
    val_persons   = persons[N_TRAIN_PERSONS:N_TRAIN_PERSONS + N_VAL_PERSONS]       # люди для val
    test_persons  = persons[N_TRAIN_PERSONS + N_VAL_PERSONS:
                            N_TRAIN_PERSONS + N_VAL_PERSONS + N_TEST_PERSONS]     # люди для test

    print("train persons:", train_persons)
    print("val persons  :", val_persons)
    print("test persons :", test_persons)

    # генерируем пары (POS/NEG) отдельно для train/val/test
    X_train, Y_train = make_pairs_from_persons(imgs, train_persons, TOTAL_POS_PER_PERSON, seed=SEED)
    X_val,   Y_val   = make_pairs_from_persons(imgs, val_persons,   TOTAL_POS_PER_PERSON, seed=SEED + 1)
    X_test,  Y_test  = make_pairs_from_persons(imgs, test_persons,  TOTAL_POS_PER_PERSON, seed=SEED + 2)

    print(f"pairs train: {X_train.shape[0]}  (pos={X_train.shape[0]//2}, neg={X_train.shape[0]//2})")
    print(f"pairs val  : {X_val.shape[0]}    (pos={X_val.shape[0]//2}, neg={X_val.shape[0]//2})")
    print(f"pairs test : {X_test.shape[0]}   (pos={X_test.shape[0]//2}, neg={X_test.shape[0]//2})")

    input_dim = X_train.shape[2:]  # форма одной картинки: (H, W, 1)

    # siamese: два входа -> общий base_network -> distance
    img_a = Input(shape=input_dim)                      # вход A
    img_b = Input(shape=input_dim)                      # вход B

    base_network = build_base_network(input_dim)        # CNN -> embedding
    feat_a = base_network(img_a)                        # embedding(A)
    feat_b = base_network(img_b)                        # embedding(B)

    dist = Lambda(euclidean_distance)([feat_a, feat_b]) # distance между embedding(A) и embedding(B)

    # модель учится уменьшать dist для POS и увеличивать для NEG (contrastive loss)
    model = Model([img_a, img_b], dist)
    model.compile(
        optimizer=RMSprop(learning_rate=LR),            # оптимизатор
        loss=contrastive_loss(margin=MARGIN)            # контрастивный лосс
    )

    # раздельные входы для fit: X[:,0]=A, X[:,1]=B
    Xtr0, Xtr1 = X_train[:, 0], X_train[:, 1]
    Xv0,  Xv1  = X_val[:, 0],   X_val[:, 1]
    Xte0, Xte1 = X_test[:, 0],  X_test[:, 1]

    model.fit(
        [Xtr0, Xtr1], Y_train,                          # обучаем на train парах
        validation_data=([Xv0, Xv1], Y_val),             # считаем val_loss на val парах
        batch_size=BATCH_SIZE,                           # размер батча
        epochs=EPOCHS,                                   # число эпох
        verbose=2                                        # вывод по эпохам
    )

    # подбираем порог по VAL и оцениваем на TEST
    pred_val = model.predict([Xv0, Xv1], verbose=0)      # distances на val
    best_t, best_acc_val = find_best_threshold(pred_val, Y_val)  # лучший threshold по val

    pred_test = model.predict([Xte0, Xte1], verbose=0)   # distances на test
    acc_best = compute_accuracy(pred_test, Y_test, threshold=best_t)  # test acc при best_t

    print(f"\nDISTANCE THRESHOLD (best_t) = {best_t:.6f}")
    print(f"RULE: dist < {best_t:.6f} => SAME, else DIFF")

    img_path = os.path.join(OUT_DIR, "inference_pairs.png")
    save_inference_grid(X_test, pred_test, best_t, img_path, num_pairs=10, seed=SEED)
    print("Saved:", img_path)

    print(f"\nVAL best_t={best_t:.6f} | VAL acc={best_acc_val:.4f}")
    print(f"TEST acc@best_t={acc_best:.4f}")

if __name__ == "__main__":
    main()