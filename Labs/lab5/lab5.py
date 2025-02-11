import numpy as np

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
    return [astar, ier,count]

def newton(f,fp,p0,tol,Nmax):
    """
    Newton iteration.
    Inputs:
    f,fp - function and derivative
    p0 - initial guess for root
    tol - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
    Returns:
    p - an array of the iterates
    pstar - the last iterate
    info - success message
    - 0 if we met tol
    - 1 if we hit Nmax iterations (fail)
    """
    p = np.zeros(Nmax+1)
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]


def combined(f,fp,a,b,btol,ntol,Nmax):
    # this will converge better than either method alone
    # still has the con that it must change sign
    [astar, ier,count] = bisection(f,a,b,btol)
    [p,pstar,info,it] = newton(f,fp,astar,ntol,Nmax)
    return [pstar,info,count+it]
def driver():
    f = lambda x: np.e**(x**2 + 7 *x -30) -1
    fp = lambda x: np.e**(x**2 + 7 *x -30) * (2*x+7)
    btol = 10 ** -2
    ntol = 10 ** -4
    ctol1 = 10 ** -2
    ctol2 = 10 ** -3
    [astar, ier,count] = bisection(f,2,4.5,btol)
    [p1,pstar1,info1,it1] = newton(f,fp,4.5,ntol,100)
    [pstar2,info2,num] = combined(f,fp,2,4.5, ctol1,ctol2,100)
    print("Bisection:", astar, "Iteration:", count)
    a ='Yes' if info1 == 0 else 'No'
    b ='Yes' if info2 == 0 else 'No' 
    print("Newton:", pstar1, "Succses:", a, "Iterations:",it1)
    print("Hybrid:", pstar2, "Succses:",b, "Iterations:",num)
driver()