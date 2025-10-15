#Отсортировать значения массива по частоте встречания. (длина массива при этом должна остаться прежней)

import numpy as np

a = np.array([4, 1, 2, 2, 3, 1, 2, 4, 4, 4, 3])

vals, counts = np.unique(a, return_counts=True)
freq = dict(zip(vals, counts))

a_sorted = sorted(a, key=lambda x: (-freq[x], x))

print(np.array(a_sorted))
