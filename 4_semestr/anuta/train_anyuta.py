import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from openwakeword.utils import AudioFeatures
from scipy.io import wavfile
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tqdm import tqdm

WINDOW_FRAMES = 16
FEAT_DIM = 96


class SimpleFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(WINDOW_FRAMES * FEAT_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_wav(path: Path, target_sr: int = 16000) -> np.ndarray:
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype != np.int16:
        data = data.astype(np.int16)
    if sr != target_sr:
        raise ValueError(f"Bad sample rate {sr} in {path}")
    return data


def clip_to_window(feats: np.ndarray, window: int) -> np.ndarray:
    if feats.shape[0] <= window:
        pad = window - feats.shape[0]
        if pad > 0:
            feats = np.pad(feats, ((0, pad), (0, 0)))
        return feats[:window]
    start = random.randint(0, feats.shape[0] - window)
    return feats[start : start + window]


def prepare_dataset(pos_dir: Path, neg_dir: Path, feats: AudioFeatures):
    X = []
    y = []

    for label, d in [(1, pos_dir), (0, neg_dir)]:
        for wav in sorted(d.glob("*.wav")):
            audio = load_wav(wav)
            emb = feats.get_embeddings(audio)  # (T, 96)
            window = clip_to_window(emb, WINDOW_FRAMES)
            X.append(window.astype(np.float32))
            y.append(label)

    X = np.stack(X)
    y = np.array(y, dtype=np.float32)
    return X, y


def train_model(X, y, device):
    # split 90/10
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.9 * len(X))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train = torch.from_numpy(X[train_idx]).to(device)
    y_train = torch.from_numpy(y[train_idx]).to(device)
    X_val = torch.from_numpy(X[val_idx]).to(device)
    y_val = torch.from_numpy(y[val_idx]).to(device)

    model = SimpleFCN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(8):
        model.train()
        opt.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val).item()
        print(f"epoch {epoch+1} loss={loss.item():.4f} val_loss={val_loss:.4f}")

    return model, (X_val.cpu().numpy(), y_val.cpu().numpy())


def compute_threshold(y_true, logits):
    probs = 1 / (1 + np.exp(-logits))
    fpr, tpr, thr = roc_curve(y_true, probs)
    # pick threshold with fpr <= 1%
    target = 0.01
    candidates = [(t, f) for t, f in zip(thr, fpr) if f <= target]
    if candidates:
        thr_sel = max(candidates, key=lambda x: x[0])[0]
    else:
        thr_sel = 0.5
    return float(thr_sel), probs


def export_onnx(model, out_path: Path):
    model.eval()
    dummy = torch.zeros(1, WINDOW_FRAMES, FEAT_DIM, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    data_dir = Path(args.data)
    pos_dir = data_dir / "pos"
    neg_dir = data_dir / "neg"
    if not pos_dir.exists() or not neg_dir.exists():
        raise FileNotFoundError("data/pos and data/neg are required")

    os.makedirs(args.out, exist_ok=True)

    feats = AudioFeatures()
    X, y = prepare_dataset(pos_dir, neg_dir, feats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, (X_val, y_val) = train_model(X, y, device)

    with torch.no_grad():
        val_logits = model(torch.from_numpy(X_val).to(device)).cpu().numpy()

    roc = roc_auc_score(y_val, 1 / (1 + np.exp(-val_logits)))
    pr = average_precision_score(y_val, 1 / (1 + np.exp(-val_logits)))
    thr, _ = compute_threshold(y_val, val_logits)

    print(f"ROC AUC={roc:.4f} PR AUC={pr:.4f} threshold={thr:.4f}")

    out_dir = Path(args.out)
    export_onnx(model, out_dir / "anyuta.onnx")

    with open(out_dir / "anyuta_threshold.txt", "w", encoding="utf-8") as f:
        f.write(f"{thr}\n")
    with open(out_dir / "label.txt", "w", encoding="utf-8") as f:
        f.write("anyuta\n")

    print("Saved models to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
