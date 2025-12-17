import torch
import torch.nn as nn


# =========================================================
# 1) MLP: 256 -> 64 -> 16 -> 4, активации ReLU, tanh, Softmax
# =========================================================

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 16),
            nn.Tanh(),

            nn.Linear(16, 4),
            nn.Softmax(dim=1)  # dim=1 потому что (batch, classes)
        )

    def forward(self, x):
        # x: (B, 256)
        return self.model(x)


# =========================================================
# 2) CNN: (B,3,19,19) -> (B,8,18,18) -> (B,8,9,9)
#         -> (B,16,8,8) -> (B,16,4,4)
# =========================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv1: 3 -> 8, kernel=2, stride=1, padding=0
        # 19 -> 18
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=2, stride=1, padding=0)
        self.act1  = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 18 -> 9

        # Conv2: 8 -> 16, kernel=2, stride=1, padding=0
        # 9 -> 8
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=1, padding=0)
        self.act2  = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8 -> 4

    def forward(self, x):
        # x: (B, 3, 19, 19)
        x = self.pool1(self.act1(self.conv1(x)))  # -> (B, 8, 9, 9)
        x = self.pool2(self.act2(self.conv2(x)))  # -> (B, 16, 4, 4)
        return x


# =========================================================
# 3) Объединение: CNN -> flatten -> Linear(256->...) как в п.1
#    Итог: (B,3,19,19) -> (B,4)
# =========================================================

class CNNPlusMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = SimpleCNN()

        # после CNN: (B,16,4,4) => 16*4*4 = 256
        self.mlp = nn.Sequential(
            nn.Flatten(),          # (B,256)
            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 16),
            nn.Tanh(),

            nn.Linear(16, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn(x)   # (B,16,4,4)
        x = self.mlp(x)   # (B,4)
        return x


# =========================================================
# Быстрая проверка форм
# =========================================================
if __name__ == "__main__":
    B = 5
    img = torch.randn(B, 3, 19, 19)

    net = CNNPlusMLP()
    out = net(img)

    print("Input :", img.shape)   # (5, 3, 19, 19)
    print("Output:", out.shape)   # (5, 4)
    print("Row sums (Softmax):", out.sum(dim=1))  # должны быть ~1
