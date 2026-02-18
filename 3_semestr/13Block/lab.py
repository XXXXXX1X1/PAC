import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# =========================
# 0) Устройство
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =========================
# 1) Данные (загрузка MNIST)
# =========================
train_dataset = torchvision.datasets.MNIST(
    root="./MNIST",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
test_dataset = torchvision.datasets.MNIST(
    root="./MNIST",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

print("=== 0. Данные ===")
print("train size:", len(train_dataset))
print("test  size:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# =========================
# 2) Модель: Conv -> Conv -> FC, активации ReLU, ReLU
#    Softmax не делаем, потому что CrossEntropyLoss ждёт logits
# =========================
class MNIST3LayerNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))   # (B,8,28,28)
        x = self.pool(x)                # (B,8,14,14)
        x = self.relu2(self.conv2(x))   # (B,16,14,14)
        x = x.view(x.size(0), -1)       # (B,3136)
        logits = self.fc(x)             # (B,10)
        return logits

model = MNIST3LayerNet().to(device)

# =========================
# 3) Loss + Optim
# =========================
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# =========================
# 4) Обучение
# =========================
print("Обучение")
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        opt.step()

        running_loss += loss.item() * y.size(0)

        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()

    train_loss = running_loss / total
    train_acc = 100.0 * correct / total
    print(f"Эпоха {epoch+1}, loss={train_loss:.4f}, accuracy={train_acc:.2f}%")

# =========================
# 5) Тест
# =========================
print("\nТестирование на новых данных:\n")
model.eval()
test_loss_sum = 0.0
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        test_loss_sum += loss.item() * y.size(0)

        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()

test_loss = test_loss_sum / total
test_acc = 100.0 * correct / total
print(f"Точность на тестовых данных: {test_acc:.2f}% | test_loss={test_loss:.4f}")
