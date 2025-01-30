import numpy as np
A = 1/2*np.array([[1,1], [1 + (10^(-10)) , 1 - (10^(-10)) ]])
def cond(A):
    eig = np.linalg.eig(A)
    mx = max(eig[0])
    mn = min(eig[0])
    return mx/mn

print(cond(A))

def relErr():
    # currently returns abs err make it rel
    abserr = lambda db1,db2: np.vstack(np.array([(10**10 +1)*db1 - (10**10) * db2, -(10**10 +1)*db1 + (10**10) * db2]))
    db1 = 10**(-5)
    db2 = 10** (-5)
    return np.vstack(abserr(db1,db2))

print(relErr())