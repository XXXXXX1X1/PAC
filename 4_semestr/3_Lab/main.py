import os
import copy, random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image


DATA_DIR = "/Users/xxx/Desktop/Учеба/Python/Pac/4_semestr/3_Lab/EuroSAT_RGB"
OUT_DIR = "./out_vis"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# ФЛАГИ РЕЖИМА
# =========================
LOAD_ONLY = True  # False = обучаем и сохраняем; True = грузим веса и только считаем
CKPT_PATH = os.path.join(OUT_DIR, "best_arcface_resnet18.pth")

# ===== Обучение / модель =====
BATCH_SIZE = 64
EPOCHS = 5
EMB_DIM = 256
LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42

# ===== Разбиение данных =====
HOLDOUT_RATIO = 0.2
VAL_IN_HOLDOUT = 0.5

# ===== ArcFace / стабильность обучения =====
MARGIN_M = 0.4  # угловая маржа ArcFac
SCALE_S = 32.0  # масштаб logits для CrossEntropy
CLIP_NORM = 5.0 # защита от большого градиента
NUM_WORKERS = 2 # параллельная загрузка данных

SHOW_PAIRS = 16
TSNE_MAX_POINTS = 3000


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_tfm():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ОЦЕНКА ТОЧНОСТИ (accuracy) НА TEST
@torch.no_grad()
def eval_acc(loader, net, device):
    net.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        emb = net.embed(x)
        cosine = net.cosine_logits(emb)
        pred = (cosine * SCALE_S).argmax(1)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return correct / total


# ArcFace: cosine -> cos(theta+m) для true класса
def arcface_logits_stable(cosine, y, m, s):
    cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)  # важно для стабильности

    cos_m = float(np.cos(m))
    sin_m = float(np.sin(m))

    sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-7))
    phi = cosine * cos_m - sine * sin_m          # cos(theta+m)

    logits = cosine.clone()
    idx = torch.arange(cosine.size(0), device=cosine.device)
    logits[idx, y] = phi[idx, y]                 # маржа только для true класса
    return logits * s


# МОДЕЛЬ: ResNet18 -> embedding -> cosine logits
class ArcFaceNet(nn.Module):
    def __init__(self, emb_dim, num_classes, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None  # pretrained веса ImageNet
        net = models.resnet18(weights=weights)                                   # backbone ResNet18

        net.conv1.stride = (1, 1)        # меньше даунсемплинга для 64x64
        net.maxpool = nn.Identity()      # убираем maxpool (сохраняем детали)

        in_feats = net.fc.in_features
        net.fc = nn.Identity()           # убираем стандартный классификатор ResNet
        self.backbone = net              # извлечение признаков

        self.fc = nn.Linear(in_feats, emb_dim)                   # проекция в embedding размером emb_dim
        self.W = nn.Parameter(torch.empty(emb_dim, num_classes))  # обучаемые веса классов для cosine logits
        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))            # инициализация W

    def embed(self, x):
        x = self.backbone(x)            # признаки из изображения
        x = self.fc(x)                  # embedding
        return F.normalize(x, dim=1)    # нормируем embedding (для ArcFace)

    def cosine_logits(self, emb):
        return emb @ F.normalize(self.W, dim=0)  # cosine similarity с весами классов -> logits


# t-SNE: собираем embeddings и сохраняем PNG
@torch.no_grad()
def collect_embeddings(loader, net, device):
    net.eval()
    embs, ys = [], []
    for x, y in loader:
        x = x.to(device)
        embs.append(net.embed(x).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(embs, 0), np.concatenate(ys, 0)

def plot_tsne_save(embs, ys, class_names, title, save_path):
    # t-SNE: уменьшаем размерность эмбеддингов (N,D) -> (N,2) для визуализации
    z = TSNE(
        n_components=2,        # хотим 2D координаты
        init="pca",            # старт через PCA
        learning_rate="auto",  # авто-шаг
        perplexity=30,         # сколько "соседей" учитывать
        random_state=SEED
    ).fit_transform(embs)

    plt.figure(figsize=(8, 6))  # создаём фигуру

    # рисуем точки разными цветами по классам
    for c, name in enumerate(class_names):
        m = (ys == c)  # маска точек класса c
        plt.scatter(z[m, 0], z[m, 1], s=8, alpha=0.7, label=name)

    plt.title(title)                                  # заголовок
    plt.legend(markerscale=2, fontsize=8, ncol=2)     # легенда
    plt.tight_layout()                                # чтобы ничего не обрезалось
    plt.savefig(save_path, dpi=200)                   # сохраняем PNG
    plt.close()                                       # закрываем фигуру (не копим память)


# INFERENCE: пары + distance
def visualize_pairs_save(net, base_ds, subset_indices, tfm, device, class_names, num_pairs, save_path):
    by_class = {}
    for idx in subset_indices:
        y = base_ds.targets[idx]
        by_class.setdefault(y, []).append(idx)

    classes = [c for c in by_class.keys() if len(by_class[c]) >= 2]
    if len(classes) < 2:
        print("Not enough samples/classes for pairs.")
        return

    pairs = []
    n_same = num_pairs // 2
    n_diff = num_pairs - n_same

    for _ in range(n_same):
        c = random.choice(classes)
        i1, i2 = random.sample(by_class[c], 2)
        pairs.append((i1, i2, 1))

    for _ in range(n_diff):
        c1, c2 = random.sample(classes, 2)
        i1 = random.choice(by_class[c1])
        i2 = random.choice(by_class[c2])
        pairs.append((i1, i2, 0))

    net.eval()
    rows = len(pairs)
    plt.figure(figsize=(8, 3 * rows))

    for r, (i1, i2, true_same) in enumerate(pairs, start=1):
        p1, y1 = base_ds.samples[i1]
        p2, y2 = base_ds.samples[i2]

        im1 = Image.open(p1).convert("RGB")
        im2 = Image.open(p2).convert("RGB")

        x1 = tfm(im1).unsqueeze(0).to(device)
        x2 = tfm(im2).unsqueeze(0).to(device)

        with torch.no_grad():
            e1 = net.embed(x1)
            e2 = net.embed(x2)
            dist = torch.norm(e1 - e2, p=2, dim=1).item()

        ax1 = plt.subplot(rows, 2, 2*r - 1)
        ax1.imshow(im1); ax1.axis("off")
        ax1.set_title(f"A: {class_names[y1]}")

        ax2 = plt.subplot(rows, 2, 2*r)
        ax2.imshow(im2); ax2.axis("off")
        ax2.set_title(f"B: {class_names[y2]}\ntrue_same={true_same} | L2 dist={dist:.4f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    seed_all(SEED)
    device = pick_device()
    print("device:", device)

    ds = datasets.ImageFolder(DATA_DIR, transform=get_tfm())
    C = len(ds.classes)

    print(f"DATASET: total_images={len(ds)} | num_classes={C}")
    print("classes:", ds.classes)

    targets = np.array(ds.targets, dtype=np.int64)
    idx_all = np.arange(len(ds), dtype=np.int64)

    train_idx, holdout_idx = train_test_split(
        idx_all, test_size=HOLDOUT_RATIO, random_state=SEED, stratify=targets
    )
    val_idx, test_idx = train_test_split(
        holdout_idx, test_size=1 - VAL_IN_HOLDOUT, random_state=SEED, stratify=targets[holdout_idx]
    )

    train_ds = Subset(ds, train_idx.tolist())  # train-часть датасета по индексам
    val_ds   = Subset(ds, val_idx.tolist())    # val-часть датасета по индексам
    test_ds  = Subset(ds, test_idx.tolist())   # test-часть датасета по индексам

    g = torch.Generator().manual_seed(SEED)    # фиксируем shuffle для воспроизводимости

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, generator=g)             # батчи train (с перемешиванием)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)                          # батчи val (без перемешивания)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)                          # батчи test (без перемешивания)

    net = ArcFaceNet(EMB_DIM, C, pretrained=True).to(device)                    # создаём модель и переносим на device

    # =========================
    # РЕЖИМ: ТОЛЬКО ЗАГРУЗКА ВЕСОВ (СЧИТАЕМ ТОЛЬКО TEST)
    # =========================
    if LOAD_ONLY:
        state = torch.load(CKPT_PATH, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:  # поддержка старого формата чекпоинта
            state = state["state_dict"]

        net.load_state_dict(state)  # применяем веса к модели
        net.eval()
        print("Loaded weights:", CKPT_PATH)

        test_acc = eval_acc(test_loader, net, device)
        print(f"TEST acc={test_acc*100:.2f}%")

        # ---- t-SNE (save only)
        test_embs, test_ys = collect_embeddings(test_loader, net, device)
        if test_embs.shape[0] > TSNE_MAX_POINTS:
            rng = np.random.default_rng(SEED)
            sel = rng.choice(test_embs.shape[0], size=TSNE_MAX_POINTS, replace=False)
            test_embs = test_embs[sel]
            test_ys = test_ys[sel]

        tsne_path = os.path.join(OUT_DIR, "tsne_test.png")
        plot_tsne_save(test_embs, test_ys, ds.classes, "EuroSAT ArcFace embeddings (t-SNE, test)", tsne_path)

        # ---- pairs (save only)
        pairs_path = os.path.join(OUT_DIR, "pairs_distance.png")
        visualize_pairs_save(net, ds, test_idx.tolist(), ds.transform, device, ds.classes, SHOW_PAIRS, pairs_path)

        print(f"\nSaved: {tsne_path}")
        print(f"Saved: {pairs_path}")
        return

    # =========================
    # РЕЖИМ: ОБУЧЕНИЕ
    # =========================
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) # оптимизатор для обучения

    best_val = -1.0
    best_state = None

    # ======= ЦИКЛ ОБУЧЕНИЯ ПО ЭПОХАМ =======
    for ep in range(1, EPOCHS + 1):
        net.train()  # включаем режим train
        total = 0  # сколько примеров обработали за эпоху
        correct = 0  # сколько угадали (для train accuracy)
        total_loss = 0.0  # суммарный loss по эпохе (для среднего train_loss)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            emb = net.embed(x)  # получаем embedding
            cosine = net.cosine_logits(emb)  # cosine logits: [B, C] (cos между emb и весами классов)

            logits_arc = arcface_logits_stable(cosine, y, MARGIN_M, SCALE_S)

            # CrossEntropy по ArcFace-логитам: хотим, чтобы истинный класс выигрывал даже с "штрафом" маржи
            loss = F.cross_entropy(logits_arc, y)

            opt.zero_grad(set_to_none=True)  # обнуляем градиенты перед backward
            loss.backward()  # считаем градиенты параметров по текущему батчу
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_NORM)
            opt.step()  # шаг оптимизатора: обновляем веса модели

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            pred = (cosine * SCALE_S).argmax(1)  # предсказанный класс
            correct += (pred == y).sum().item()  # сколько угадали в батче

        train_loss = total_loss / total
        train_acc = correct / total
        val_acc = eval_acc(val_loader, net, device)

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(net.state_dict())

        print(f"ep {ep:02d}/{EPOCHS} | loss={train_loss:.4f} | train={train_acc*100:.2f}% | val={val_acc*100:.2f}%")

    net.load_state_dict(best_state)

    torch.save(net.state_dict(), CKPT_PATH)
    print("Saved weights:", CKPT_PATH)

    test_acc = eval_acc(test_loader, net, device)
    print(f"\nBEST val={best_val*100:.2f}% | FINAL test={test_acc*100:.2f}%")

    # ---- t-SNE (save only)
    test_embs, test_ys = collect_embeddings(test_loader, net, device)
    if test_embs.shape[0] > TSNE_MAX_POINTS:
        rng = np.random.default_rng(SEED)
        sel = rng.choice(test_embs.shape[0], size=TSNE_MAX_POINTS, replace=False)
        test_embs = test_embs[sel]
        test_ys = test_ys[sel]

    tsne_path = os.path.join(OUT_DIR, "tsne_test.png")
    plot_tsne_save(test_embs, test_ys, ds.classes, "EuroSAT ArcFace embeddings (t-SNE, test)", tsne_path)

    # ---- pairs (save only)
    pairs_path = os.path.join(OUT_DIR, "pairs_distance.png")
    visualize_pairs_save(net, ds, test_idx.tolist(), ds.transform, device, ds.classes, SHOW_PAIRS, pairs_path)

    print(f"\nSaved: {tsne_path}")
    print(f"Saved: {pairs_path}")


if __name__ == "__main__":
    main()