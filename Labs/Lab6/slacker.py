import numpy as np
from numpy import random as rand
import math
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from matplotlib import cm

def slacker(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0 #initial guess
    rn = x0 #list of iterates
    Fn = f(xn) #function value vector
    # compute n x n Jacobian matrix (ONLY ONCE)
    Jn = Jf(xn)
    # Use pivoted LU factorization to solve systems for Jf. Makes lusolve O(n^2)
    lu, piv = lu_factor(Jn)

    n=0
    nf=1 
    nJ=1 #function and Jacobian evals
    npn=1

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|")

    while npn>tol and n<=nmax:
   
        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)))

        # Newton step (we could check whether Jn is close to singular here)
        
        
        pn = -lu_solve((lu, piv), Fn)#We use lu solve instead of pn = -np.linalg.solve(Jn,Fn);
        # if(abs(xn[0]+pn[0]) > abs(xn[0]-pn[0])):
       #    Jn = Jf(xn)
            # Use pivoted LU factorization to solve systems for Jf. Makes lusolve O(n^2)
      #     lu, piv = lu_factor(Jn)
        xn = xn + pn
        
        npn = np.linalg.norm(pn) #size of Newton step

        n+=1
        rn = np.vstack((rn,xn))
        Fn = f(xn)
        nf+=1

    r=xn

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Lazy Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)))
        else:
            print("Lazy Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)))

    return (r,rn,nf,nJ)

def f(x):
    [x1,x2] = x
    return [4*x1**2+x2**2-4,x1+x2-np.sin(x1-x2)]

def driver():
    # this performed similiarly to my lab partner
    x0 = [1,0]
    j11 = lambda x: 8*x[0]
    j12 = lambda x : 2*x[1]
    j21 = lambda x: 1 -np.cos(x[0]-x[1])
    j22 = lambda x: 1 + np.cos(x[0]-x[1])
    Jf = lambda x: np.array([[j11(x),j12(x)],[j21(x),j22(x)]])
    [r,rn,nf,nJ] = slacker(f,Jf,x0,10**-10,100,verb=False)
    print(r)
driver()