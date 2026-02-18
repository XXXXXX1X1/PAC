import numpy as np

A = np.array([[1,0,1],
              [0,1,0],
              [1,0,1]], dtype=float)

U, S, Vt = np.linalg.svd(A, full_matrices=True)

print("Сингулярные значения S:", S)  # ~ [2., 1., 0.]
print("матрица левых сингулярных векторов=\n", U)
print("транспонированная матрица правых сингулярных векторов=\n", Vt)

