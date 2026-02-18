import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('dino.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

template = image[675:725, 70:115].copy()
plt.figure(figsize=(1, 1))
plt.imshow(template)
plt.show()


match = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
plt.imshow(match)#, cmap='hot')
plt.colorbar()
plt.show()

np.where(match == match.max()) # координаты "мэтча"

# define model
model = torchvision.models.resnet18(pretrained=True)
layer4_features = None
avgpool_emb = None

# define hooks
def get_features(module, inputs, output):
    global layer4_features
    layer4_features = output

def get_embedding(module, inputs, output):
    global avgpool_emb
    avgpool_emb = output

model.layer4.register_forward_hook(get_features)
model.avgpool.register_forward_hook(get_embedding)
model.eval()