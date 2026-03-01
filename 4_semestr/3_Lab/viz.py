import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from train import (
    DATA_DIR, SEED, BATCH_SIZE, NUM_WORKERS, CKPT_PATH, SPLIT_PATH,
    pick_device, get_tfm, ArcFaceNet
)

MAX_TSNE = 5000
SAVE_TSNE = "tsne_test.png"
SAVE_SAME = "pair_same.png"
SAVE_DIFF = "pair_diff.png"

def denorm(img_chw: torch.Tensor) -> np.ndarray:
    x = img_chw.detach().cpu()
    x = (x * 0.5 + 0.5).clamp(0, 1)
    return x.permute(1, 2, 0).numpy()

def save_pair(img1, img2, title, out_path):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1); plt.imshow(denorm(img1)); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(denorm(img2)); plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved:", out_path)

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = pick_device()
    print("device:", device)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    class_names = ckpt["class_names"]
    emb_dim = ckpt["emb_dim"]
    C = len(class_names)

    split = np.load(SPLIT_PATH)
    test_idx = split["test_idx"]

    ds = datasets.ImageFolder(DATA_DIR, transform=get_tfm())
    targets = np.array(ds.targets, dtype=np.int64)

    test_ds = Subset(ds, test_idx.tolist())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    net = ArcFaceNet(emb_dim, C).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    # t-SNE
    embs, labs, got = [], [], 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            embs.append(net.embed(x).cpu().numpy())
            labs.append(y.numpy())
            got += x.size(0)
            if got >= MAX_TSNE:
                break

    test_embs = np.concatenate(embs, 0)[:MAX_TSNE]
    test_labs = np.concatenate(labs, 0)[:MAX_TSNE]
    n = test_embs.shape[0]
    print("tsne points:", test_embs.shape)

    if n >= 10:
        perplexity = min(30, max(5, (n - 1) // 3))
        xy = TSNE(2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=SEED).fit_transform(test_embs)
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(xy[:, 0], xy[:, 1], c=test_labs, s=6)
        plt.colorbar(sc)
        plt.title(f"EuroSAT test embeddings (t-SNE), perplexity={perplexity}")
        plt.tight_layout()
        plt.savefig(SAVE_TSNE, dpi=200)
        plt.close()
        print("saved:", SAVE_TSNE)

    # pairs
    cls_idx = {c: [] for c in range(C)}
    for si, bi in enumerate(test_ds.indices):
        cls_idx[int(targets[bi])].append(si)

    have_pairs = [c for c in range(C) if len(cls_idx[c]) >= 2]
    non_empty = [c for c in range(C) if len(cls_idx[c]) > 0]
    if not have_pairs or len(non_empty) < 2:
        print("pairs skipped: not enough samples")
        return

    c = random.choice(have_pairs)
    i1, i2 = random.sample(cls_idx[c], 2)
    img1, _ = test_ds[i1]
    img2, _ = test_ds[i2]
    with torch.no_grad():
        e1 = net.embed(img1.unsqueeze(0).to(device))[0].cpu()
        e2 = net.embed(img2.unsqueeze(0).to(device))[0].cpu()
    d_same = float(1.0 - F.cosine_similarity(e1[None], e2[None]).item())
    save_pair(img1, img2, f"SAME {class_names[c]} | cosine_dist={d_same:.4f}", SAVE_SAME)

    c1, c2 = random.sample(non_empty, 2)
    img1, _ = test_ds[random.choice(cls_idx[c1])]
    img2, _ = test_ds[random.choice(cls_idx[c2])]
    with torch.no_grad():
        e1 = net.embed(img1.unsqueeze(0).to(device))[0].cpu()
        e2 = net.embed(img2.unsqueeze(0).to(device))[0].cpu()
    d_diff = float(1.0 - F.cosine_similarity(e1[None], e2[None]).item())
    save_pair(img1, img2, f"DIFF {class_names[c1]} vs {class_names[c2]} | cosine_dist={d_diff:.4f}", SAVE_DIFF)

if __name__ == "__main__":
    main()