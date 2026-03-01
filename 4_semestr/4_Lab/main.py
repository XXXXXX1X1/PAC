import os, random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from sklearn.decomposition import PCA


# ---------- CONFIG ----------
SEED = 42
DATA_DIR = "./data"
OUT_DIR = "./out_ae"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH = 128
LR = 1e-3
EPOCHS = 30
LATENT = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN = (DEVICE == "cuda")

labeldict = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
}


# ---------- SEED ----------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(SEED)


# ---------- DATA ----------
tfm = transforms.ToTensor()
trainset = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tfm)
testset  = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tfm)
train_loader = DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=PIN)
test_loader  = DataLoader(testset,  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=PIN)


# ---------- MODELS ----------
class FCAE(nn.Module):
    def __init__(self, latent=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(),
                                 nn.Linear(128, 256), nn.ReLU(),
                                 nn.Linear(256, 784), nn.Sigmoid())

    def forward(self, x):
        b = x.size(0)
        z = self.enc(x.view(b, -1))
        out = self.dec(z).view(b, 1, 28, 28)
        return out

    def encode(self, x):
        b = x.size(0)
        return self.enc(x.view(b, -1))

    def decode(self, z):
        return self.dec(z).view(z.size(0), 1, 28, 28)


class ConvAE(nn.Module):
    def __init__(self, latent=4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),   # 14x14
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU()   # 7x7
        )
        self.enc_fc = nn.Linear(64 * 7 * 7, latent)

        self.dec_fc = nn.Linear(latent, 64 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),  # 14x14
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid() # 28x28
        )

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        return self.enc_fc(h)

    def decode(self, z):
        h = self.dec_fc(z).view(z.size(0), 64, 7, 7)
        return self.dec(h)


# ---------- HELPERS ----------
def save_loss(losses, path, title):
    plt.figure()
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

@torch.no_grad()
def save_recon(model, loader, path, n=16):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(DEVICE)
    r = model(x)

    grid_in  = make_grid(x.cpu(), nrow=8, padding=2)
    grid_out = make_grid(r.cpu(), nrow=8, padding=2)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.title("Input"); plt.imshow(grid_in.permute(1,2,0).squeeze(), cmap="gray"); plt.axis("off")
    plt.subplot(1, 2, 2); plt.title("Recon"); plt.imshow(grid_out.permute(1,2,0).squeeze(), cmap="gray"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

@torch.no_grad()
def save_latent_pca(model, loader, path, max_points=3000, title="latent PCA"):
    model.eval()
    Z, Y = [], []
    got = 0
    for x, y in loader:
        x = x.to(DEVICE)
        z = model.encode(x).cpu().numpy()
        Z.append(z); Y.append(y.numpy())
        got += x.size(0)
        if got >= max_points:
            break
    Z = np.concatenate(Z)[:max_points]
    Y = np.concatenate(Y)[:max_points]

    xy = PCA(n_components=2).fit_transform(Z)
    plt.figure(figsize=(10,7))
    plt.title(title)
    sc = plt.scatter(xy[:,0], xy[:,1], c=Y, s=8)
    plt.colorbar(sc)
    for i in range(10):
        m = xy[Y == i].mean(axis=0)
        plt.text(m[0], m[1], f"{labeldict[i]} ({i})", fontsize=9, ha="center")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def train(model, out_prefix, epochs=30, noise_std=0.0):
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    losses = []

    for ep in range(1, epochs+1):
        model.train()
        s = 0.0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            x_in = x
            if noise_std > 0:
                x_in = (x + noise_std * torch.randn_like(x)).clamp(0, 1)

            opt.zero_grad(set_to_none=True)
            r = model(x_in)
            loss = loss_fn(r, x)
            loss.backward()
            opt.step()
            s += loss.item() * x.size(0)

        l = s / len(train_loader.dataset)
        losses.append(l)
        print(f"[{out_prefix}] {ep:03d}/{epochs} loss={l:.6f}")

    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{out_prefix}.pth"))
    save_loss(losses, os.path.join(OUT_DIR, f"{out_prefix}_loss.png"), f"{out_prefix} loss")
    save_recon(model, test_loader, os.path.join(OUT_DIR, f"{out_prefix}_recon.png"))
    save_latent_pca(model, test_loader, os.path.join(OUT_DIR, f"{out_prefix}_latent_pca.png"), title=f"{out_prefix} PCA")
    return model


# ---------- RUN ----------
fc = train(FCAE(LATENT), f"fc_ae_lat{LATENT}", epochs=EPOCHS, noise_std=0.0)
cae = train(ConvAE(LATENT), f"conv_ae_lat{LATENT}", epochs=EPOCHS, noise_std=0.0)

print("done ->", OUT_DIR)

