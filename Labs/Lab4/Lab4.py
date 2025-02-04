import numpy as np

def fixedpt(f,x0,tol,Nmax):
    count = 0
    x = np.array(Nmax,1)
    while (count <Nmax):
        count = count +1
        x1 = f(x0)
        x[count] = x1
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            return [xstar,ier]
        x0 = x1
    xstar = x1
    ier = 1
    return [x, ier]

def a(p,pvec):
    
