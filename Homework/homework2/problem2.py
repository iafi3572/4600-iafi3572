import numpy as np
A = 1/2*np.array([[1,1], [1 + (10^(-10)) , 1 - (10^(-10)) ]])
def cond(A):
    eig = np.linalg.eig(A)
    mx = max(eig[0])
    mn = min(eig[0])
    return mx/mn

print("Condition Number:",cond(A))

def relErr(db1,db2):
    # currently returns abs err make it rel
    diff = lambda db1,db2: np.vstack(np.array([(10**10 +1)*db1 - (10**10) * db2, -(10**10 +1)*db1 + (10**10) * db2]))
    abserr = np.linalg.norm(diff(db1,db2))

    return abserr/np.linalg.norm(np.vstack(np.array([1,1])))

db1 = 1.1*10**(-4)
db2 = 10** (-4)
print("Perturbation: \n ", 'Db1:' ,db1,'\n  Db2:',db2, )
print("Relative Error:",relErr(db1,db2))