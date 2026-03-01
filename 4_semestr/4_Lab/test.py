import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch
ds = datasets.FashionMNIST("./data", train=True, download=False, transform=transforms.ToTensor())

# берём первые 16
imgs = torch.stack([ds[i][0] for i in range(16)])  # (16,1,28,28)
labels = [ds[i][1] for i in range(16)]

grid = make_grid(imgs, nrow=4, padding=2)

plt.figure(figsize=(6, 6))
plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
plt.title("FashionMNIST: 16 samples")
plt.axis("off")
plt.show()

print("labels:", labels)