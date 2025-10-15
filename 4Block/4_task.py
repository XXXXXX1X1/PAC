import numpy as np


A = np.array([[3,4,5],[1,2,3],[2,2,3],[5,7,1]], dtype=float)

s = A.sum(axis=1)          # сумма по строке
m = A.max(axis=1)          # максимальный элемент в строке
mask = (m < (s - m)) & np.all(A > 0, axis=1)  # неравенства треугольника и положительность
triangles = A[mask]
print(triangles)
