import torch
import torch.nn as nn


# =========================================================
# 1) MLP через nn.Sequential(): 256 -> 64 -> 16 -> 4
#    Активации: ReLU, Tanh, Softmax
# =========================================================
class MLPSequential(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 16),
            nn.Tanh(),

            nn.Linear(16, 4),
            nn.Softmax(dim=1)  # dim=1: (batch, classes)
        )

    def forward(self, x):
        # x: (B, 256)
        return self.model(x)


# =========================================================
# 2) MLP через nn.Module (явно слоями): 256 -> 64 -> 16 -> 4
# =========================================================
class MLPModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)

        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (B, 256)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


# =========================================================
# 3) CNN через nn.Module:
#    (B,3,19,19) -> (B,8,18,18) -> (B,8,9,9)
#               -> (B,16,8,8)  -> (B,16,4,4)
# =========================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 19 -> 18 (kernel=2, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=2, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # 18 -> 9
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 9 -> 8
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # 8 -> 4
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # x: (B, 3, 19, 19)
        x = self.pool1(self.relu1(self.conv1(x)))  # -> (B, 8, 9, 9)
        x = self.pool2(self.relu2(self.conv2(x)))  # -> (B, 16, 4, 4)
        return x


# =========================================================
# 4) Объединение: CNN -> Flatten -> MLP (256 -> 64 -> 16 -> 4)
#    Итог: (B,3,19,19) -> (B,4)
# =========================================================
class CNNPlusMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = SimpleCNN()

        self.head = nn.Sequential(
            nn.Flatten(),          # (B,16,4,4) -> (B,256)
            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 16),
            nn.Tanh(),

            nn.Linear(16, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn(x)    # (B,16,4,4)
        x = self.head(x)   # (B,4)
        return x


# =========================================================
# Быстрая проверка форм (как в задании)
# =========================================================
if __name__ == "__main__":
    B = 5

    # Проверка MLP
    vec = torch.randn(B, 256)
    mlp_seq = MLPSequential()
    mlp_mod = MLPModule()
    out1 = mlp_seq(vec)
    out2 = mlp_mod(vec)

    print("MLPSequential input :", vec.shape, "output:", out1.shape, "row_sums:", out1.sum(dim=1))
    print("MLPModule     input :", vec.shape, "output:", out2.shape, "row_sums:", out2.sum(dim=1))

    # Проверка CNN + объединение
    img = torch.randn(B, 3, 19, 19)
    cnn = SimpleCNN()
    comb = CNNPlusMLP()

    feat = cnn(img)
    out3 = comb(img)

    print("CNN input  :", img.shape,  "output:", feat.shape)  # (B,16,4,4)
    print("Combo input:", img.shape,  "output:", out3.shape, "row_sums:", out3.sum(dim=1))
