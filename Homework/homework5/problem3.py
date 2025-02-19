import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
def fixed(G,x0,tol,nmax):

    # Initialize arrays and function value
    xn = x0 #initial guess
    rn = x0 #list of iterates
    Gn = x0 - np.dot(np.array([[1/6,1/18],[0,1/6]]),G(xn)) #function value vector
    n=0
    nf=1  #function evals

    while np.linalg.norm(Gn-xn)>tol and n<=nmax:
        xn = Gn
        n+=1
        rn = np.vstack((rn,xn))
        Gn = xn - np.dot(np.array([[1/6,1/18],[0,1/6]]),G(xn))
        nf+=1

        if np.linalg.norm(xn)>1e15:
            n=nmax+1
            nf=nmax+1
            break
    r=xn
    return (r,rn,n)