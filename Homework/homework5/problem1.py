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

def evalJ(x):
    J = np.array([[6*x[0],-2*x[1]],[3*x[1]**2-3*x[0]**2, 6*x[0]*x[1]]])
    return J

def evalF(x0):
    f = lambda x,y: 3*x**2 - y **2
    g = lambda x,y: 3*x*y**2 - x**3 -1
    return (np.array([f(x0[0],x0[1]),g(x0[0],x0[1])])).transpose()

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

def driver():
    x0 = np.array([1,1])
    tol = 10**-5
    [r,rn,n] = fixed(evalF,x0,tol,100)
    [xstar,ier,its] = Newton(x0,tol,10000)
    print("FPT")
    print("\tSolution: [x,y] =",r)
    print("\tIterations",n)
    print("Newton")
    print("\tSolution: [x,y] =",xstar)
    print("\tIterations:",its)
driver()