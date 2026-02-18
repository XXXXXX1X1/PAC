#Решить следующую систему линейных уравнений (не используя np.linalg.solve):


import numpy as np

A = np.array([[3, 4, 2],
              [5, 2, 3], [4, 3, 2]], dtype=float)

b = np.array([17, 23, 19], dtype=float)

x = np.linalg.inv(A) @ b
print(x)
