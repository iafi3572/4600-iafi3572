import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def newtons(f,fp,x0,tol,Nmax):
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

def newton2c(f,fp,x0,tol,Nmax,m):
    x = np.zeros(Nmax+1)
    x[0] = x0
    for i in range(Nmax):
        x1 = x0 - m*f(x0)/fp(x0)
        x[i+1] = x1
        if(abs(x1-x0) < tol):
            xstar = x1
            info = 0
            return [x,xstar,info,i]
        x0 =x1
    xstar = x1
    info = 1
    return [x,xstar,info,i]

def modnewton(f,fp,fpp,x0,tol,Nmax):
    newf = lambda x: f(x)/fp(x)
    newfp = lambda x: ((fp(x)*fp(x)) - (f(x)*fpp(x)))/(fp(x)**2)
    x = np.zeros(Nmax+1)
    x[0] = x0
    for i in range(Nmax):
        x1 = x0 - newf(x0)/newfp(x0)
        x[i+1] = x1
        if(abs(x1-x0) < tol):
            xstar = x1
            info = 0
            return [x,xstar,info,i]
        x0 =x1
    xstar = x1
    info = 1
    return [x,xstar,info,i]

def driver():
    x0 = 4
    tol = 10**-4
    Nmax = 200
    m = 2
    f = lambda x : np.exp(3*x) + 27*(-x**6+ x**4*np.exp(x)) - 9 * x**2 * np.exp(2*x)
    fp= lambda x : 3*np.exp(3*x) - (162 * x**5) + (108 * x**3 * np.exp(x)) + (27*np.exp(x)* x **4) - (18*x*np.exp(2*x)) - (18 * x**2*np.exp(2*x))
    fpp = lambda x: 9 * np.exp(3*x) - 810 * x**4 + (324 * x**2 * np.exp(x)) + (216* x**3 *np.exp(x)) + (27*np.exp(x)*x**4) -18*np.exp(2*x) - (72 * x * np.exp(2*x)) - (36*x**2 *np.exp(2*x))
    [x1,xstar1,info1,i1] = newtons(f,fp,x0,tol,Nmax)
    [x2,xstar2,info2,i2] = newton2c(f,fp,x0,tol,Nmax,m)
    [x3,xstar3,info3,i3] = modnewton(f,fp,fpp,x0,tol,Nmax)
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(abs(x1[0:i1-1]-xstar1), abs(x1[1:i1]-xstar1),label = "Newton")
    plt.plot(abs(x2[0:i2-1]-xstar2), abs(x2[1:i2]-xstar2), label = "2c")
    plt.plot(abs(x3[0:i3-1]-xstar3), abs(x3[1:i3]-xstar3),label = "Class")
    plt.legend()
    
    print("Newton:")
    print("\tRoot:", xstar1) 
    print("\tSuccsess:","Yes" if info1 ==0 else "No")
    print("\tOrder:",linregress(np.log(abs(x1[0:i1-1]-xstar1)), np.log(abs(x1[1:i1]-xstar1)))[0])
    print("Problem 2c")
    print("\tRoot:", xstar2) 
    print("\tSuccsess:","Yes" if info2 ==0 else "No")
    print("\tOrder:",linregress(np.log(abs(x2[0:i2-1]-xstar2)), np.log(abs(x2[1:i2]-xstar2)))[0])
    print("Class")
    print("\tRoot:", xstar3) 
    print("\tSuccsess:", "Yes"if info3 ==0 else "No")
    print("\tClass Order:",linregress(np.log(abs(x3[0:i3-1]-xstar3)), np.log(abs(x3[1:i3]-xstar3)))[0])
    plt.savefig("plot4.png")
driver()
