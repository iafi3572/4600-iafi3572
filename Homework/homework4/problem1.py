import numpy as np
import scipy as sp
import scipy.special as spec
import scipy.integrate as integrate 
import matplotlib.pyplot as plt
t = 86400 * 60
alpha = .138 * 10 **-6
Ti = 20
Ts = -15
def f(x):
    return spec.erf(x/(2*np.sqrt(alpha * t))) * (Ti -Ts) + Ts
def fp(x):
    c = (Ti-Ts)/(np.sqrt(alpha*t*np.pi))
    return c * np.exp(-x**2/(4*alpha*t))

def newton(f,fp,x0,tol,Nmax):
    x = np.zeros(Nmax+1)
    x[0] = x0
    for i in range(Nmax):
        if (fp(x0) == 0):
            xstar = x0
            return(x,xstar,"No",i)
        x1 = x0 - f(x0)/fp(x0)
       
        x[i+1] = x1
        if(abs(x1-x0) < tol):
            xstar = x1
            info = 0
            return [x,xstar,"Yes",i]
        x0 =x1
    xstar = x1
    info = 1
    return [x,xstar,"Yes",i]

def bisection(f,a,b,tol):
    fa = f(a)
    fb = f(b)
    if(fa*fb > 0):
        astar = a
        ier =1
        return [astar,ier]
    if (fa == 0):
        astar = a
        ier =0
        return [astar, ier]
    if (fb ==0):
        astar = b
        ier = 0
        return [astar, ier]
    count = 0
    d = 0.5*(a+b)
    while(abs(d-a) > tol):
        fd = f(d)
        if(fd ==0):
            astar = d
            ier = 0
            return [astar, ier]
        if (fa*fd<0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5*(a+b)
        count = count +1
    astar = d
    ier = 0
    return [astar, ier]
def driver():
    tol = 10 **-13
    xend = 2.5
    x = np.linspace(0,xend)
   
    plt.plot(x,f(x))
    plt.ylabel("f(x)")
    plt.savefig("plot1.png")
    dBi = bisection(f,0,xend,tol)[0]
    [_,dNe1,info1,_] = newton(f,fp,.01,tol,200)
    [_,dNe2,info2,_] = newton(f,fp,xend,tol,200)
    print("Bisection:", dBi)
    print("Newton, x0 = .01:", dNe1, "Succsess:", info1)
    print(f"Newton, x0 = {xend}: {dNe2} Succsess: {info2}")
driver()