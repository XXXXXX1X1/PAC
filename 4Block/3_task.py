#Написать функцию, вычисляющую плавающее среднее вектора

import numpy as np

def movav(arr, win):
    kernel = np.ones(win) / win

    return np.convolve(arr, kernel, mode='valid')

x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
print(movav(x, 3))
