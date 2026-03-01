import os
import numpy as np
import imageio.v2 as imageio

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop

# -----------------------
# Конфиг эксперимента
# -----------------------
DATA_DIR = "/Users/xxx/Desktop/Учеба/Python/Pac/4_semestr/2_Lab/orl_faces"
SEED = 42

# Деление по людям (без утечки): train/val/test содержат разные личности
N_TRAIN_PERSONS = 20
N_VAL_PERSONS = 10
N_TEST_PERSONS = 10

# Сколько положительных пар генерируем на одного человека
TOTAL_POS_PER_PERSON = 250

BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
MARGIN = 1.0


# -----------------------
# Загружаем ORL один раз: (40 людей × 10 фото × H × W)
# -----------------------
def load_orl_images(data_dir: str) -> np.ndarray:
    imgs = []
    for person in range(1, 41):       # папки s1..s40
        row = []
        for idx in range(1, 11):      # файлы 1..10.pgm
            path = os.path.join(data_dir, f"s{person}", f"{idx}.pgm")
            img = imageio.imread(path)  # (H, W) uint8
            row.append(img)
        imgs.append(row)
    return np.array(imgs, dtype=np.uint8)


# -----------------------
# Генерация пар только внутри заданных person_ids (чтобы не было утечки)
# X: (N, 2, H, W, 1) — пары изображений
# Y: (N, 1) — метка: 1 = один человек, 0 = разные
# -----------------------
def make_pairs_from_persons(imgs: np.ndarray, person_ids, total_pos_per_person: int, seed: int):
    rng = np.random.default_rng(seed)

    person_ids = np.array(person_ids, dtype=np.int32)
    n_persons = len(person_ids)
    n_imgs = imgs.shape[1]
    H, W = imgs.shape[2], imgs.shape[3]

    total_pos = n_persons * total_pos_per_person
    total_neg = total_pos  # балансируем классы: POS = NEG
    total = total_pos + total_neg

    X = np.empty((total, 2, H, W, 1), dtype=np.uint8)
    Y = np.empty((total, 1), dtype=np.float32)

    # POS: два разных снимка одного и того же человека
    k = 0
    for pid in person_ids:
        for _ in range(total_pos_per_person):
            a, b = rng.choice(n_imgs, size=2, replace=False)
            X[k, 0, :, :, 0] = imgs[pid, a]
            X[k, 1, :, :, 0] = imgs[pid, b]
            Y[k, 0] = 1.0
            k += 1

    # NEG: снимки двух разных людей
    for _ in range(total_neg):
        p1, p2 = rng.choice(person_ids, size=2, replace=False)
        a = rng.integers(0, n_imgs)
        b = rng.integers(0, n_imgs)
        X[k, 0, :, :, 0] = imgs[p1, a]
        X[k, 1, :, :, 0] = imgs[p2, b]
        Y[k, 0] = 0.0
        k += 1

    # Перемешивание + нормализация пикселей в [0, 1]
    idx = rng.permutation(total)
    X = X[idx].astype(np.float32) / 255.0
    Y = Y[idx]
    return X, Y


# -----------------------
# Базовая CNN: превращает изображение в embedding (вектор признаков)
# Общая для обеих веток (shared weights)
# -----------------------
def build_base_network(input_shape):
    return Sequential([
        Input(shape=input_shape),

        Conv2D(6, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(12, (3, 3), activation="relu", padding="valid"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.1),
        Dense(50, activation="relu"),
    ])


# Евклидова дистанция между двумя embedding (чем меньше, тем больше сходство)
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


# Contrastive loss:
# POS (y=1): тянем distance -> 0
# NEG (y=0): раздвигаем distance >= margin
def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        return K.mean(
            y_true * K.square(y_pred) +
            (1.0 - y_true) * K.square(K.maximum(margin - y_pred, 0.0))
        )
    return loss


# Точность по порогу: distance < threshold => "один человек"
def compute_accuracy(distances, labels, threshold=0.5):
    preds_same = (distances.ravel() < threshold).astype(np.float32)
    return float(np.mean(preds_same == labels.ravel()))


# Подбор лучшего порога по VAL (чтобы не подгонять threshold по TEST)
def find_best_threshold(distances, labels):
    d = distances.ravel()
    candidates = np.quantile(d, np.linspace(0.01, 0.99, 99))
    best_t, best_acc = float(candidates[0]), -1.0
    for t in candidates:
        acc = compute_accuracy(distances, labels, threshold=float(t))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def main():
    # Фиксируем сиды для воспроизводимости
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Загружаем все изображения
    imgs = load_orl_images(DATA_DIR)  # (40, 10, H, W)

    # Делим людей на train/val/test (разные люди -> без утечки)
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

    # Генерируем пары отдельно для train/val/test (без пересечений людей)
    X_train, Y_train = make_pairs_from_persons(imgs, train_persons, TOTAL_POS_PER_PERSON, seed=SEED)
    X_val, Y_val = make_pairs_from_persons(imgs, val_persons, TOTAL_POS_PER_PERSON, seed=SEED + 1)
    X_test, Y_test = make_pairs_from_persons(imgs, test_persons, TOTAL_POS_PER_PERSON, seed=SEED + 2)

    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_val  :", X_val.shape, "Y_val  :", Y_val.shape)
    print("X_test :", X_test.shape, "Y_test :", Y_test.shape)

    input_dim = X_train.shape[2:]  # (H, W, 1)

    # Siamese: два входа -> общий base_network -> два embedding -> дистанция
    img_a = Input(shape=input_dim)
    img_b = Input(shape=input_dim)

    base_network = build_base_network(input_dim)
    feat_a = base_network(img_a)
    feat_b = base_network(img_b)

    dist = Lambda(euclidean_distance)([feat_a, feat_b])

    # Обучаем модель минимизировать contrastive loss
    model = Model([img_a, img_b], dist)
    model.compile(
        optimizer=RMSprop(learning_rate=LR),
        loss=contrastive_loss(margin=MARGIN)
    )

    model.fit(
        [X_train[:, 0], X_train[:, 1]],
        Y_train,
        validation_data=([X_val[:, 0], X_val[:, 1]], Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2
    )

    # Порог подбираем по VAL, затем финально оцениваем на TEST
    pred_val = model.predict([X_val[:, 0], X_val[:, 1]], verbose=0)
    best_t, best_acc_val = find_best_threshold(pred_val, Y_val)
    print("\nVAL лучший порог:", best_t, "VAL точность:", best_acc_val)

    pred_test = model.predict([X_test[:, 0], X_test[:, 1]], verbose=0)
    acc_05 = compute_accuracy(pred_test, Y_test, threshold=0.5)
    acc_best = compute_accuracy(pred_test, Y_test, threshold=best_t)

    print("\nTEST точность (порог=0.5):", acc_05)
    print("TEST точность (порог=best_t из VAL):", acc_best)


if __name__ == "__main__":
    main()