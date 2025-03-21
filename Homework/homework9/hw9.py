import numpy as np

A = np.array([[581, 105],[105,454]])
b = np.array([421,247])

print(np.linalg.solve(A,b))