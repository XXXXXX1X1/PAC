import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights


DATA_ROOT = "/Users/xxx/Desktop/Учеба/Python/Pac/4_semestr/2_Lab/orl_faces"
NUM_TRAIN_PERSONS = 27
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
MARGIN = 1.0


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_images(folder):
    imgs = []
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if os.path.isfile(p) and f.lower().endswith(".pgm"):
            imgs.append(p)
    imgs.sort()
    return imgs


def create_pairs(root, num_train_persons):
    folders = [os.path.join(root, f) for f in os.listdir(root)
               if os.path.isdir(os.path.join(root, f)) and f.startswith("s")]
    folders.sort()
    random.shuffle(folders)

    train_people = {}
    test_people = {}

    for idx, folder in enumerate(folders):
        imgs = list_images(folder)[:10]
        if len(imgs) < 2:
            continue
        if idx < num_train_persons:
            train_people[idx] = imgs
        else:
            test_people[idx] = imgs

    def build_pairs(people_dict):
        ids = sorted(people_dict.keys())
        pairs = []

        # positive pairs
        for i in ids:
            imgs = people_dict[i]
            for a in range(len(imgs)):
                for b in range(a + 1, len(imgs)):
                    pairs.append([imgs[a], imgs[b], 1])

        # negative pairs (по 1 паре на пару людей)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, k = ids[a], ids[b]
                pairs.append([people_dict[i][0], people_dict[k][0], 0])

        random.shuffle(pairs)
        return pairs

    return build_pairs(train_people), build_pairs(test_people)


class FacePairsDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        self.t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, y = self.pairs[idx]
        x1 = self.t(Image.open(p1).convert("RGB"))
        x2 = self.t(Image.open(p2).convert("RGB"))
        return x1, x2, torch.tensor(y, dtype=torch.float32)


class Siamese(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # (B,512,1,1)
        self.fc = nn.Linear(resnet.fc.in_features, 128)
        self.margin = margin

    def encode(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        return self.encode(x1), self.encode(x2)

    def loss(self, z1, z2, y):
        d = F.pairwise_distance(z1, z2)        # (B,)
        loss_same = y * 0.5 * (d ** 2)
        loss_diff = (1 - y) * 0.5 * (torch.clamp(self.margin - d, min=0.0) ** 2)
        return (loss_same + loss_diff).mean()


@torch.no_grad()
def test_accuracy(model, loader, device, thr=None):
    model.eval()

    # простой порог = середина средних расстояний same/diff
    if thr is None:
        ds, dd = [], []
        for x1, x2, y in loader:
            x1, x2 = x1.to(device), x2.to(device)
            y = y.to(device).view(-1)
            z1, z2 = model(x1, x2)
            d = F.pairwise_distance(z1, z2).view(-1)
            ds.append(d[y == 1].cpu().numpy())
            dd.append(d[y == 0].cpu().numpy())
        ds = np.concatenate([a for a in ds if len(a) > 0])
        dd = np.concatenate([a for a in dd if len(a) > 0])
        thr = 0.5 * (float(ds.mean()) + float(dd.mean()))

    correct = 0
    total = 0
    for x1, x2, y in loader:
        x1, x2 = x1.to(device), x2.to(device)
        y = y.to(device).view(-1)
        z1, z2 = model(x1, x2)
        d = F.pairwise_distance(z1, z2).view(-1)
        pred = (d < thr).float()
        correct += (pred == y).sum().item()
        total += y.numel()

    return 100.0 * correct / total, thr


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = get_device()
    print("device:", device)

    train_pairs, test_pairs = create_pairs(DATA_ROOT, NUM_TRAIN_PERSONS)

    train_loader = DataLoader(FacePairsDataset(train_pairs), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(FacePairsDataset(test_pairs),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = Siamese(margin=MARGIN).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x1, x2, y in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            y = y.to(device).view(-1)

            opt.zero_grad()
            z1, z2 = model(x1, x2)
            loss = model.loss(z1, z2, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc, thr = test_accuracy(model, test_loader, device, thr=None)
        print(f"Epoch {epoch+1}/{EPOCHS} | loss={avg_loss:.4f} | test_acc={acc:.2f}% | thr={thr:.4f}")


if __name__ == "__main__":
    main()