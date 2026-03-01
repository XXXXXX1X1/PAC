from torchvision import datasets

datasets.FashionMNIST(root="./data", train=True, download=True)
datasets.FashionMNIST(root="./data", train=False, download=True)

print("Done. Saved to ./data")