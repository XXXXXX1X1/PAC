import numpy as np

A = np.array([[2, 1],
              [5, 7]], dtype=float)

b = np.array([11, 13], dtype=float)

x = np.linalg.pinv(A) @ b
print(x)
