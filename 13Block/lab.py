import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =========================================================
# 3 слоя (learnable): Conv -> Conv -> FC
# активации: ReLU, ReLU, Softmax
# =========================================================
class MNIST3LayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1) Conv: (B,1,28,28) -> (B,8,28,28) (padding=1 чтобы размер не менялся)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        # 2) Conv: (B,8,28,28) -> (B,16,28,28)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # Чтобы FC не был гигантским, делаем downsample через pooling (не обучаемый слой)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        # 3) FC: (B,16,14,14) -> (B,10)
        self.fc = nn.Linear(16 * 14 * 14, 10)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool(x)                   # (B,8,14,14)
        x = self.relu2(self.conv2(x))
        x = x.reshape(x.size(0), -1)       # flatten -> (B, 16*14*14)
        x = self.fc(x)                     # logits -> (B,10)
        x = self.softmax(x)                # probs  -> (B,10)
        return x


# =========================================================
# Loss: Cross-Entropy по вероятностям (раз Softmax обязателен)
# L = -mean(log(p_true_class))
# =========================================================
def ce_from_probs(probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eps = 1e-9
    probs = torch.clamp(probs, eps, 1.0)
    return -torch.log(probs[torch.arange(probs.size(0), device=probs.device), target]).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        probs = model(x)
        loss = ce_from_probs(probs, y)

        pred = probs.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        total_loss += loss.item() * y.size(0)

    return total_loss / total, correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # MNIST: (0..1) -> нормализация (стандартно для MNIST)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = MNIST3LayerNet().to(device)

    # Оптимизатор
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()

            probs = model(x)
            loss = ce_from_probs(probs, y)

            loss.backward()
            opt.step()

            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

        train_loss = running_loss / seen
        test_loss, test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | test_acc={test_acc*100:.2f}%")

    # пример предсказания
    model.eval()
    x0, y0 = test_ds[0]
    with torch.no_grad():
        p0 = model(x0.unsqueeze(0).to(device)).cpu()[0]
    print("\nExample:")
    print("true =", int(y0), "| pred =", int(torch.argmax(p0)), "| probs =", p0.numpy())


if __name__ == "__main__":
    main()
