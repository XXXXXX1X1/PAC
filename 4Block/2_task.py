#Дана картинка высоты - h, ширины - w, количество каналов - 3, тип данных np.uint8 (от 0 до 255).
# Найти количество уникальных цветов (один цвет это три числа [122, 0, 255]).



import numpy as np
def unique_rgb_count(img: np.ndarray):
    flat = img.reshape(-1, 3)
    return np.unique(flat, axis=0).shape[0]

# мини-пример
img = np.array([[[255,0,0],[255,0,0],[0,255,0]],
                [[0,255,0],[0,0,255],[0,0,255]],
                [[0,255,0],[0,0,255],[0,0,255]]], dtype=np.uint8)
print(unique_rgb_count(img))  # 3
