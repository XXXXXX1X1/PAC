import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt


# -------------------------
# 1) INPUTS
# -------------------------
template_path = "dino2.png"
search_path = "dino.jpg"

y1, y2 = 480, 560
x1, x2 = 30, 230

temp_rgb = cv2.cvtColor(cv2.imread(template_path), cv2.COLOR_BGR2RGB)
img_rgb = cv2.cvtColor(cv2.imread(search_path), cv2.COLOR_BGR2RGB)
template = temp_rgb[y1:y2, x1:x2]


# -------------------------
# 2) MODEL + HOOKS
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

layer4_features = None  # сюда hook положит output layer4
avgpool_emb = None  # сюда hook положит output avgpool


def hook_layer4(module, inputs, output):
    global layer4_features
    layer4_features = output


def hook_avgpool(module, inputs, output):
    global avgpool_emb
    avgpool_emb = output


model = (
    torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    .to(device)
    .eval()
)

# регистрируем hooks
model.layer4.register_forward_hook(hook_layer4)
model.avgpool.register_forward_hook(hook_avgpool)

# -------------------------
# 3) ImageNet normalization
# -------------------------
mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]


def prep(rgb: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(device) / 255.0
    return (x - mean) / std


# -------------------------
# 4) EXTRACT FEATURES VIA HOOKS
# -------------------------
with torch.no_grad():
    # прогоняем template -> получаем avgpool_emb (q)
    _ = model(prep(template))
    q = avgpool_emb.flatten()  # (1,512,1,1) -> (512,)

    # прогоняем full image -> получаем layer4_features (fm)
    _ = model(prep(img_rgb))
    fm = layer4_features.squeeze(0)  # (1,512,h,w) -> (512,h,w)

# -------------------------
# 5) cosine similarity heatmap
# -------------------------
q = F.normalize(q, dim=0)  # нормируем вектор запроса
fm = F.normalize(fm, dim=0)  # нормируем каждый вектор fm[:,h,w]

heat = torch.einsum("c,chw->hw", q, fm).cpu().numpy()

H, W = img_rgb.shape[:2]
heat_up = cv2.resize(heat, (W, H))

y_max, x_max = np.unravel_index(np.argmax(heat_up), heat_up.shape)
print(x_max, y_max)

# -------------------------
# 6) VISUALIZATION (3 окна)
# -------------------------
fig, ax = plt.subplots(1, 3, figsize=(16, 5))

ax[0].imshow(template)
ax[0].set_title("Template (query)")
ax[0].axis("off")

ax[1].imshow(img_rgb)
ax[1].scatter([x_max], [y_max], marker="x", s=140)
ax[1].set_title("Search image")
ax[1].axis("off")

ax[2].imshow(img_rgb)
ax[2].imshow(heat_up, cmap="jet", alpha=0.45)
ax[2].scatter([x_max], [y_max], marker="x", s=140)
ax[2].set_title("Heatmap (cosine similarity)")
ax[2].axis("off")

plt.tight_layout()
plt.show()
