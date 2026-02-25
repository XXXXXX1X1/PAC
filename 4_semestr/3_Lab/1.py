import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from pytorch_metric_learning.losses import ArcFaceLoss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "/Users/xxx/Desktop/Учеба/Python/Pac/4_semestr/3_Lab/EuroSAT_RGB"  # <-- путь до папки с 10 классами
BATCH_SIZE = 64          # для Mac CPU/MPS обычно лучше 32-64
EPOCHS = 10
EMB_DIM = 128
LR = 1e-3
SEED = 42
TEST_RATIO = 0.2

PRINT_EVERY = 20         # печать прогресса каждые N шагов
NUM_WORKERS = 0          # macOS safe (spawn). Можно 2, если всё обернуто в __main__ (у нас обернуто)
PIN_MEMORY = False       # для MPS не нужно, для CUDA можно True


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    # Приоритет: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -----------------------
# MODEL: backbone -> embedding
# -----------------------
class EmbedNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=None)  # можно weights="IMAGENET1K_V1" если разрешено
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.fc = nn.Linear(in_feats, emb_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = F.normalize(x, dim=1)  # важно для ArcFace
        return x


def cosine_distance(a, b) -> float:
    return float(1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def denorm_img(img_chw: torch.Tensor) -> np.ndarray:
    # img in normalized tensor (C,H,W) -> numpy (H,W,C) in [0,1]
    x = img_chw.detach().cpu().clone()
    x = x * 0.5 + 0.5
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def show_pair(img1, img2, title: str):
    i1 = denorm_img(img1)
    i2 = denorm_img(img2)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(i1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(i2)
    plt.axis("off")

    plt.suptitle(title)
    plt.show()


@torch.no_grad()
def collect_embeddings(model, loader, device):
    model.eval()
    embs = []
    labs = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        emb = model(imgs).cpu().numpy()
        embs.append(emb)
        labs.append(labels.numpy())
    return np.concatenate(embs), np.concatenate(labs)


@torch.no_grad()
def embedding_of(model, img_tensor, device):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)
    emb = model(img_tensor)[0].cpu()
    return emb


def build_class_index(subset):
    """
    subset: torch.utils.data.Subset(ImageFolder)
    returns dict[class_id] -> list of subset indices (0..len(subset)-1)
    """
    base = subset.dataset
    idxs = subset.indices
    m = {}
    for si, bi in enumerate(idxs):
        _, y = base[bi]
        m.setdefault(y, []).append(si)
    return m


def main():
    seed_all(SEED)

    device = pick_device()
    print("device:", device)

    # (опционально) чуть ускоряет CPU на некоторых системах
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    # -----------------------
    # DATA
    # -----------------------
    tfm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    full_ds = datasets.ImageFolder(DATA_DIR, transform=tfm)
    class_names = full_ds.classes
    num_classes = len(class_names)

    print("classes:", class_names)
    print("total images:", len(full_ds))

    n_test = int(len(full_ds) * TEST_RATIO)
    n_train = len(full_ds) - n_test

    train_ds, test_ds = random_split(
        full_ds,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )

    # На macOS multiprocessing DataLoader = боль, но с __main__ уже можно пробовать 2
    # Если вдруг снова будут ошибки spawn — верни NUM_WORKERS=0
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Быстрая проверка, что DataLoader живой (и чтобы не было "тишины")
    imgs0, labels0 = next(iter(train_loader))
    print("first batch:", tuple(imgs0.shape), tuple(labels0.shape))

    # -----------------------
    # MODEL + ARCFACE
    # -----------------------
    model = EmbedNet(EMB_DIM).to(device)

    # ArcFaceLoss хранит веса классов внутри, поэтому его параметры тоже оптимизируем
    loss_fn = ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=EMB_DIM,
        margin=0.4,
        scale=32
    ).to(device)

    opt = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=LR
    )

    # ускорители: cudnn autotune (только cuda)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # -----------------------
    # TRAIN LOOP
    # -----------------------
    def train_one_epoch():
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for step, (imgs, labels) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=False)
            labels = labels.to(device, non_blocking=False)

            emb = model(imgs)
            loss = loss_fn(emb, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

            logits = loss_fn.get_logits(emb)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()

            if step % PRINT_EVERY == 0:
                print(f"  step {step}/{len(train_loader)} | loss={loss.item():.4f}")

        return total_loss / total, correct / total

    @torch.no_grad()
    def eval_acc(loader):
        model.eval()
        total = 0
        correct = 0
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            emb = model(imgs)
            logits = loss_fn.get_logits(emb)
            pred = logits.argmax(dim=1)
            total += imgs.size(0)
            correct += (pred == labels).sum().item()
        return correct / total

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch()
        te_acc = eval_acc(test_loader)
        print(f"epoch {ep:02d}/{EPOCHS} | train_loss={tr_loss:.4f} | train_acc={tr_acc*100:.2f}% | test_acc={te_acc*100:.2f}%")

    # -----------------------
    # t-SNE ON TEST
    # -----------------------
    test_embs, test_labs = collect_embeddings(model, test_loader, device)
    print("test_embs:", test_embs.shape)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=SEED
    )
    xy = tsne.fit_transform(test_embs)

    plt.figure(figsize=(10, 8))
    for c in range(num_classes):
        idx = (test_labs == c)
        plt.scatter(xy[idx, 0], xy[idx, 1], s=6, label=class_names[c])
    plt.legend(markerscale=3, fontsize=8)
    plt.title("EuroSAT test embeddings (t-SNE)")
    plt.show()

    # -----------------------
    # INFERENCE PAIRS + DISTANCE
    # -----------------------
    class_index = build_class_index(test_ds)

    # same-class pair
    c = random.randrange(num_classes)
    i1, i2 = random.sample(class_index[c], 2)
    img1, _ = test_ds[i1]
    img2, _ = test_ds[i2]

    e1 = embedding_of(model, img1, device)
    e2 = embedding_of(model, img2, device)
    d_same = cosine_distance(e1, e2)
    show_pair(img1, img2, f"SAME class={class_names[c]} | cosine_dist={d_same:.4f}")

    # different-class pair
    c1, c2 = random.sample(range(num_classes), 2)
    i1 = random.choice(class_index[c1])
    i2 = random.choice(class_index[c2])
    img1, _ = test_ds[i1]
    img2, _ = test_ds[i2]

    e1 = embedding_of(model, img1, device)
    e2 = embedding_of(model, img2, device)
    d_diff = cosine_distance(e1, e2)
    show_pair(img1, img2, f"DIFF {class_names[c1]} vs {class_names[c2]} | cosine_dist={d_diff:.4f}")


if __name__ == "__main__":
    main()