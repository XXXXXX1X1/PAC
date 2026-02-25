# EuroSAT + ArcFace (STABLE) + CUDA/MPS/CPU + t-SNE(test) + inference(пары+distance)
# Идея фикса: считаем ArcFace без arccos (через cos(m), sin(m)), + grad clipping, + нормальные logits для accuracy
# pip install torch torchvision scikit-learn matplotlib pillow

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "/Users/xxx/Desktop/Учеба/Python/Pac/4_semestr/3_Lab/EuroSAT_RGB"
BATCH_SIZE = 64
EPOCHS = 15
EMB_DIM = 256

LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42
TEST_RATIO = 0.2

MARGIN_M = 0.4         # angular margin
SCALE_S = 32.0         # было 32 -> мягче и стабильнее
CLIP_NORM = 5.0        # защита от взрыва градиентов

NUM_WORKERS = 0        # macOS safe
SAVE_TSNE = "tsne_test.png"
SAVE_SAME = "pair_same.png"
SAVE_DIFF = "pair_diff.png"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -----------------------
# MODEL: encoder -> embedding
# -----------------------
class EmbedNet(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        net = models.resnet18(weights=None)
        in_feats = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.fc = nn.Linear(in_feats, emb_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = F.normalize(x, dim=1)   # нормализуем фичи
        return x


# -----------------------
# ArcFace head (cosine logits) + stable ArcFace logits (без arccos!)
# -----------------------
class CosineComponent(nn.Module):
    def __init__(self, emb_size: int, output_classes: int):
        super().__init__()
        self.output_classes = output_classes
        self.W = nn.Parameter(torch.empty(emb_size, output_classes))
        nn.init.kaiming_uniform_(self.W)

    def forward(self, x):
        # cosine = <x_norm, W_norm>
        x_norm = F.normalize(x, dim=1)
        W_norm = F.normalize(self.W, dim=0)
        return x_norm @ W_norm


def arcface_logits_stable(cosine, target, num_classes: int, m: float, s: float):
    """
    Stable ArcFace:
      phi = cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
      where sin(theta) = sqrt(1 - cos^2(theta))
    No arccos -> меньше шанс взрыва градиентов/NaN.
    """
    cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    cos_m = float(np.cos(m))
    sin_m = float(np.sin(m))

    sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-7))
    phi = cosine * cos_m - sine * sin_m  # cos(theta+m)

    one_hot = F.one_hot(target, num_classes=num_classes).to(cosine.dtype)
    logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
    logits = logits * s
    return logits


def cosine_distance(a, b) -> float:
    return float(1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def denorm(img_chw: torch.Tensor) -> np.ndarray:
    x = img_chw.detach().cpu().clone()
    x = x * 0.5 + 0.5
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def save_pair(img1, img2, title: str, out_path: str):
    i1 = denorm(img1)
    i2 = denorm(img2)
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1); plt.imshow(i1); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(i2); plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved:", out_path)


@torch.no_grad()
def collect_test_embeddings(model, loader, device):
    model.eval()
    embs, labs = [], []
    for x, y in loader:
        x = x.to(device)
        e = model(x).cpu().numpy()
        embs.append(e)
        labs.append(y.numpy())
    return np.concatenate(embs), np.concatenate(labs)


@torch.no_grad()
def embed_one(model, img, device):
    model.eval()
    return model(img.unsqueeze(0).to(device))[0].cpu()


def build_class_index(subset):
    base = subset.dataset
    idxs = subset.indices
    m = {}
    for si, bi in enumerate(idxs):
        _, y = base[bi]
        m.setdefault(y, []).append(si)
    return m


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = pick_device()
    print("device:", device)

    tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    ds = datasets.ImageFolder(DATA_DIR, transform=tfm)
    class_names = ds.classes
    C = len(class_names)
    print("classes:", class_names)
    print("total images:", len(ds))

    n_test = int(len(ds) * TEST_RATIO)
    n_train = len(ds) - n_test
    train_ds, test_ds = random_split(ds, [n_train, n_test], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = EmbedNet(EMB_DIM).to(device)
    head = CosineComponent(EMB_DIM, C).to(device)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # -----------------------
    # TRAIN
    # -----------------------
    for ep in range(1, EPOCHS + 1):
        model.train()
        head.train()

        total_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            emb = model(x)
            cosine = head(emb)

            logits = arcface_logits_stable(cosine, y, num_classes=C, m=MARGIN_M, s=SCALE_S)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), CLIP_NORM)
            opt.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()

        train_loss = total_loss / total
        train_acc = correct / total

        # TEST
        model.eval()
        head.eval()
        t_total = 0
        t_correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                emb = model(x)
                cosine = head(emb)
                logits = arcface_logits_stable(cosine, y, num_classes=C, m=MARGIN_M, s=SCALE_S)
                pred = logits.argmax(dim=1)
                t_total += x.size(0)
                t_correct += (pred == y).sum().item()
        test_acc = t_correct / t_total

        print(f"epoch {ep:02d}/{EPOCHS} | train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | test_acc={test_acc*100:.2f}%")

    # -----------------------
    # t-SNE on TEST embeddings
    # -----------------------
    test_embs, test_labs = collect_test_embeddings(model, test_loader, device)
    print("test_embs:", test_embs.shape)

    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=SEED)
    xy = tsne.fit_transform(test_embs)

    plt.figure(figsize=(10, 8))
    for c in range(C):
        idx = (test_labs == c)
        plt.scatter(xy[idx, 0], xy[idx, 1], s=6, label=class_names[c])
    plt.legend(markerscale=3, fontsize=8)
    plt.title("EuroSAT test embeddings (t-SNE)")
    plt.tight_layout()
    plt.savefig(SAVE_TSNE, dpi=200)
    plt.close()
    print("saved:", SAVE_TSNE)

    # -----------------------
    # Inference: pairs + distance
    # -----------------------
    cls_idx = build_class_index(test_ds)

    # SAME
    c = random.randrange(C)
    i1, i2 = random.sample(cls_idx[c], 2)
    img1, _ = test_ds[i1]
    img2, _ = test_ds[i2]
    e1 = embed_one(model, img1, device)
    e2 = embed_one(model, img2, device)
    d_same = cosine_distance(e1, e2)
    save_pair(img1, img2, f"SAME {class_names[c]} | cosine_dist={d_same:.4f}", SAVE_SAME)

    # DIFF
    c1, c2 = random.sample(range(C), 2)
    i1 = random.choice(cls_idx[c1])
    i2 = random.choice(cls_idx[c2])
    img1, _ = test_ds[i1]
    img2, _ = test_ds[i2]
    e1 = embed_one(model, img1, device)
    e2 = embed_one(model, img2, device)
    d_diff = cosine_distance(e1, e2)
    save_pair(img1, img2, f"DIFF {class_names[c1]} vs {class_names[c2]} | cosine_dist={d_diff:.4f}", SAVE_DIFF)


if __name__ == "__main__":
    main()