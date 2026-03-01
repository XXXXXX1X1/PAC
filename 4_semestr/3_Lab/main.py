import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "/home/alex/PycharmProjects/PAC/4_semestr/3_Lab/EuroSAT/2750"

BATCH_SIZE = 64
EPOCHS = 15
EMB_DIM = 256

LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42
TEST_RATIO = 0.2

MARGIN_M = 0.4
SCALE_S = 32.0
CLIP_NORM = 5.0

NUM_WORKERS = 2
MAX_TSNE = 5000

SAVE_TSNE = "tsne_test.png"
SAVE_SAME = "pair_same.png"
SAVE_DIFF = "pair_diff.png"


def arcface_logits_stable(cosine, y, num_classes: int, m: float, s: float):
    cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    cos_m, sin_m = float(np.cos(m)), float(np.sin(m))
    sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-7))
    phi = cosine * cos_m - sine * sin_m  # cos(theta+m)
    one_hot = F.one_hot(y, num_classes=num_classes).to(cosine.dtype)
    logits = one_hot * phi + (1.0 - one_hot) * cosine
    return logits * s


def denorm(img_chw: torch.Tensor) -> np.ndarray:
    x = img_chw.detach().cpu()
    x = (x * 0.5 + 0.5).clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def save_pair(img1, img2, title: str, out_path: str):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1); plt.imshow(denorm(img1)); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(denorm(img2)); plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved:", out_path)


class ArcFaceNet(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()
        net = models.resnet18(weights=None)
        in_feats = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.fc = nn.Linear(in_feats, emb_dim)
        self.W = nn.Parameter(torch.empty(emb_dim, num_classes))
        nn.init.kaiming_uniform_(self.W)

    def embed(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return F.normalize(x, dim=1)

    def cosine_logits(self, emb):
        Wn = F.normalize(self.W, dim=0)
        return emb @ Wn


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
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

    # -------- stratified split --------
    targets = np.array(ds.targets, dtype=np.int64)
    idx_all = np.arange(len(ds), dtype=np.int64)

    train_idx, test_idx = train_test_split(
        idx_all,
        test_size=TEST_RATIO,
        random_state=SEED,
        stratify=targets
    )

    train_ds = Subset(ds, train_idx.tolist())
    test_ds = Subset(ds, test_idx.tolist())

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, generator=g)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    net = ArcFaceNet(EMB_DIM, C).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # -----------------------
    # TRAIN
    # -----------------------
    for ep in range(1, EPOCHS + 1):
        net.train()
        total_loss, total, correct = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            emb = net.embed(x)
            cosine = net.cosine_logits(emb)

            logits_arc = arcface_logits_stable(cosine, y, num_classes=C, m=MARGIN_M, s=SCALE_S)
            loss = F.cross_entropy(logits_arc, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_NORM)
            opt.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            pred = (cosine * SCALE_S).argmax(dim=1)
            correct += (pred == y).sum().item()

        train_loss = total_loss / total
        train_acc = correct / total

        # TEST
        net.eval()
        t_total, t_correct = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                emb = net.embed(x)
                cosine = net.cosine_logits(emb)
                pred = (cosine * SCALE_S).argmax(dim=1)
                t_total += x.size(0)
                t_correct += (pred == y).sum().item()

        test_acc = t_correct / t_total
        print(f"epoch {ep:02d}/{EPOCHS} | train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | test_acc={test_acc*100:.2f}%")

    # -----------------------
    # t-SNE on TEST embeddings (subsample)
    # -----------------------
    net.eval()
    embs, labs, got = [], [], 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            e = net.embed(x).cpu().numpy()
            embs.append(e)
            labs.append(y.numpy())
            got += x.size(0)
            if got >= MAX_TSNE:
                break

    test_embs = np.concatenate(embs, axis=0)[:MAX_TSNE]
    test_labs = np.concatenate(labs, axis=0)[:MAX_TSNE]
    print("tsne points:", test_embs.shape)

    xy = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=SEED).fit_transform(test_embs)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=test_labs, s=6)
    plt.colorbar(sc)
    plt.title("EuroSAT test embeddings (t-SNE)")
    plt.tight_layout()
    plt.savefig(SAVE_TSNE, dpi=200)
    plt.close()
    print("saved:", SAVE_TSNE)

    # -----------------------
    # Inference: pairs + cosine distance
    # -----------------------
    # build per-class indices inside test_ds
    cls_idx = {c: [] for c in range(C)}
    for si, bi in enumerate(test_ds.indices):
        cls_idx[targets[bi]].append(si)

    # SAME
    c = random.randrange(C)
    while len(cls_idx[c]) < 2:
        c = random.randrange(C)
    i1, i2 = random.sample(cls_idx[c], 2)
    img1, _ = test_ds[i1]
    img2, _ = test_ds[i2]

    with torch.no_grad():
        e1 = net.embed(img1.unsqueeze(0).to(device))[0].cpu()
        e2 = net.embed(img2.unsqueeze(0).to(device))[0].cpu()
    d_same = float(1.0 - F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item())
    save_pair(img1, img2, f"SAME {class_names[c]} | cosine_dist={d_same:.4f}", SAVE_SAME)

    # DIFF
    c1, c2 = random.sample(range(C), 2)
    img1, _ = test_ds[random.choice(cls_idx[c1])]
    img2, _ = test_ds[random.choice(cls_idx[c2])]

    with torch.no_grad():
        e1 = net.embed(img1.unsqueeze(0).to(device))[0].cpu()
        e2 = net.embed(img2.unsqueeze(0).to(device))[0].cpu()
    d_diff = float(1.0 - F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item())
    save_pair(img1, img2, f"DIFF {class_names[c1]} vs {class_names[c2]} | cosine_dist={d_diff:.4f}", SAVE_DIFF)



if __name__ == "__main__":
    main()