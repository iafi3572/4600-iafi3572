import numpy as np
import scipy as sc
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt

def fixed(f,x0,tol,nmax,fp):
    d = f(x0)/(fp(x0)[0]**2 +fp(x0)[1]**2 +fp(x0)[2]**2)
    xn = x0 
    rn = x0 
    Gn = x0 - d*np.array([fp(x0)[0], fp(x0)[1], fp(x0)[2]]) #function value vector
    n=0
    nf=1  #function evals

    while np.linalg.norm(Gn-xn)>tol and n<=nmax:
        xn = Gn
        n+=1
        rn = np.vstack((rn,xn))
        d = f(xn)/(fp(xn)[0]**2 +fp(xn)[1]**2 +fp(xn)[2]**2)
        Gn = xn - d*np.array([fp(xn)[0], fp(xn)[1], fp(xn)[2]]) #function value vector
        nf+=1

        if np.linalg.norm(xn)>1e15:
            n=nmax+1
            nf=nmax+1
            break
    r=xn
    return (r,rn,n)

def fpartial(x):
    p = np.zeros(3)
    p[0] = 2*x[0] 
    p[1] = 8*x[1]
    p[0] = 8*x[2]
    return p

def f(x):
    return x[0] **2 + 4*x[1]**2 + 4 * x[2] **2 -16

def driver():
    x0 = [1,1,1]
    tol = 10**-5
    nmax = 100
    [r,rn,n] = fixed(f,x0,tol,nmax,fpartial)
    print(f'x: {r[0]}')
    print(f'y: {r[1]}') 
    print(f'z: {r[2]}')
    print(f'Iterations: {n}')
    en = (((rn -r)[0:n-1]))
    enp1 = (((rn -r)[1: n]))
    enn = np.zeros(n-1)
    ennp1 = np.zeros(n-1)
    for i in range(n-1):
        enn[i] = abs(norm(en[i]))
        ennp1[i] = abs(norm(enp1[i]))
    
    plt.plot(enn**2,ennp1)
    slope = sc.stats.linregress(np.log(enn),np.log(ennp1)).slope
    print(slope)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig("plot3.png")
driver()