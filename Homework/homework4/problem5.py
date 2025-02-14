import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from scipy.stats import linregress

def newton(f,fp,x0,tol,Nmax):
    x = np.zeros(Nmax+1)
    x[0] = x0
    for i in range(Nmax):
        x1 = x0 - f(x0)/fp(x0)
        x[i+1] = x1
        if(abs(x1-x0) < tol):
            xstar = x1
            info = 0
            return [x,xstar,info,i]
        x0 =x1
    xstar = x1
    info = 1
    return [x,xstar,info,i]

def secant(f,x0,x1,Nmax,tol):
    n=0
    x = np.zeros(Nmax+1)
    while(n<Nmax and abs(x1-x0) >=tol):
        m = (f(x1)-f(x0))/(x1-x0)
        x0=x1
        x[n] = x1
        x1 = x1 - f(x1)/m
        n = n+1
    r= x1
    return [x,r,n]


def driver():
    nmax = 200
    tol = 10 ** -5
    f = lambda x: x**6 -x -1
    fp = lambda x: 6*x**5 -1

    [x1,xstar,info,i] = newton(f,fp,2,tol,nmax)
    [x2,r,n] = secant(f,2,1,nmax,tol)

    b = max(n,i)
    step = np.arange(0,b,1)
    er1 = abs(x1[0:b]-xstar)
    er2 = abs(x2[0:b]-r)
    d = df({'Step' : step, 'Newton Error': er1,
            'Secant Error': er2})
    d.to_csv('table.txt', sep = '\t', index = False)

    er1 = abs(x1[0:b-1]-xstar)
    er1a = abs(x1[1:b]-xstar)
    er2 = abs(x2[0:b-1]-r)
    er2a = abs(x2[1:b]-r)
    plt.plot(er1,er1a,label= "Newton")
    plt.plot(er2,er2a,label= "Secant")
    print("Newtons:",linregress(np.log(er1), np.log(er1a))[0])
    print("Secant:" ,linregress(np.log(er2), np.log(er2a))[0])
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.savefig("plot5.png")

driver()


        