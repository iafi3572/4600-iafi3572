import numpy as np
u = np.array([10, 4])
e1 = np.array([1, 0])
v = u + np.sign(u[0]) * np.linalg.norm(u) * e1
v = v / np.linalg.norm(v)

H = np.eye(2) - 2 * np.outer(v, v)
Q = np.eye(3)
Q[1:, 1:] = H
A = np.array(([12, 10, 4], [10, 8, -5], [4, -5, 3]), dtype=float)
T = Q.T @ A @ Q

# Output
print("H:\n", H)
print("\nQ:\n", Q)
print("\nT:\n", T)
