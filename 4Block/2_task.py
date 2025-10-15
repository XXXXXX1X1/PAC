import numpy as np
def unique_rgb_count(img: np.ndarray):
    flat = img.reshape(-1, 3)
    print(flat)
    return np.unique(flat, axis=0).shape[0]

# мини-пример
img = np.array([[[255,0,0],[255,0,0],[0,255,0]],
                [[0,255,0],[0,0,255],[0,0,255]]], dtype=np.uint8)
print(unique_rgb_count(img))  # 3
