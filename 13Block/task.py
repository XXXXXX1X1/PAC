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
            nn.Linear(256, 64),     # (B,256) -> (B,64)
            nn.ReLU(),              # ReLU(t)=max(0,t)

            nn.Linear(64, 16),      # (B,64) -> (B,16)
            nn.Tanh(),              # tanh(t) ∈ [-1,1]

            nn.Linear(16, 4),       # (B,16) -> (B,4)
            #  превращаем в вероятности:
            # dim=1 потому что размер (B, classes), классы сидят в оси 1
            nn.Softmax(dim=1)       # (B,4) -> (B,4), сумма по строке ≈ 1
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


        self.fc1 = nn.Linear(256, 64)   # (B,256) -> (B,64)
        self.fc2 = nn.Linear(64, 16)    # (B,64)  -> (B,16)
        self.fc3 = nn.Linear(16, 4)     # (B,16)  -> (B,4)

        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.act2(x)

        x = self.fc3(x)
        x = self.softmax(x)
        return x


# =========================================================
# 3) CNN через nn.Module:
#    (B,3,19,19) -> (B,8,18,18) -> (B,8,9,9)
#               -> (B,16,8,8)  -> (B,16,4,4)
# =========================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # out = floor((in + 2*padding - kernel_size)/stride) + 1

        # conv1: 3 -> 8
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2, stride=1, padding=0)  # (B,3,19,19)->(B,8,18,18)
        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                # (B,8,18,18)->(B,8,9,9)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=0) # (B,8,9,9)->(B,16,8,8)
        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                # (B,16,8,8)->(B,16,4,4)

    def forward(self, x):


        #Conv -> ReLU -> Pool
        x = self.conv1(x)   # (B,8,18,18)
        x = self.relu1(x)   # (B,8,18,18)
        x = self.pool1(x)   # (B,8,9,9)

        # Conv -> ReLU -> Pool
        x = self.conv2(x)   # (B,16,8,8)
        x = self.relu2(x)   # (B,16,8,8)
        x = self.pool2(x)   # (B,16,4,4)

        return x


class CNNPlusMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = SimpleCNN()
        self.mlp = MLPSequential()   # берём MLP из пункта 1

    def forward(self, x):
        x = self.cnn(x)              # (B,16,4,4)
        x = torch.flatten(x, 1)      # (B,256)  flatten по всем, кроме batch
        x = self.mlp(x)              # (B,4)
        return x




if __name__ == "__main__":
    B = 5

    # --- Проверка MLP ---
    vec = torch.randn(B, 256)   # случайный батч из 5 векторов длины 256

    mlp_seq = MLPSequential()
    mlp_mod = MLPModule()

    out1 = mlp_seq(vec)         # (B,4)
    out2 = mlp_mod(vec)         # (B,4)

    # row_sums должны быть ~1 из-за Softmax
    print("MLPSequential input :", vec.shape, "output:", out1.shape, "row_sums:", out1.sum(dim=1))
    print("MLPModule     input :", vec.shape, "output:", out2.shape, "row_sums:", out2.sum(dim=1))

    # --- Проверка CNN + объединение ---
    img = torch.randn(B, 3, 19, 19)  # 5 “картинок” 3x19x19

    cnn = SimpleCNN()
    comb = CNNPlusMLP()

    feat = cnn(img)             # (B,16,4,4) признаки
    out3 = comb(img)            # (B,4) вероятности

    print("CNN input  :", img.shape,  "output:", feat.shape)  # (B,16,4,4)
    print("Combo input:", img.shape,  "output:", out3.shape, "row_sums:", out3.sum(dim=1))
