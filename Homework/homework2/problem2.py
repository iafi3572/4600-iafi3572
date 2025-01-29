import numpy as np
A = 1/2*np.array([[1,1], [1 + (10^(-10)) , 1 - (10^(-10)) ]])
def cond(A):
    eig = np.linalg.eig(A)
    mx = max(eig[0])
    mn = min(eig[0])
    return mx/mn

print(cond(A))