import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm

# wish to find roots to system f1,f2. If we can do this then we will have found a1,a2 b1,b2
def f1(a1,a2):
    return np.sqrt(1+(a1+a2)**2)/np.sqrt(2) - 2/3 - a1 
def f2(a1,a2):
    return np.sqrt(1+(a1-a2)**2)/np.sqrt(2) - 2/3 - a2

def evalJ(x):
    J = np.array([[],[]])
    return J

def evalF(x0):
    return (np.array([f1(x0[0],x0[1]),f2(x0[0],x0[1])]))

def Newton(x0,tol,Nmax):
    # inputs: x0 = initial guess, tol = tolerance, Nmax = max its
    # Outputs: xstar= approx root, ier = error message, its = num its
    for its in range(Nmax):
        J = evalJ(x0)
        F = evalF(x0)
        P = np.linalg.solve(J,F)
        x1 = x0 - P
        if (norm(x1-x0) < tol):
            xstar = x1
            ier =0
            return[xstar, ier, its]
        x0 = x1
    xstar = x1
    ier = 1
    return[xstar,ier,its]

def driver()