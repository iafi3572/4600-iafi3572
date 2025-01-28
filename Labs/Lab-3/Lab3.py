import numpy as np
f1 = lambda x : (x**2)*(x-1)
f2 = lambda x: (x-1) * (x-3) * (x-5)
f3 = lambda x : (x-1) **2 * (x-3)
f4 = lambda x :  np.sin(x)
f5 = lambda x : x *(1+ (7-x**5)/x**2)**3
f6 = lambda x : x - ((x**5 -7)/(x**2))
f7 = lambda x : x - ((x**5 - 7 )/(5* (x**4)))
f8 = lambda x : x - ((x**5 -7)/12)

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

def fixedpt(f,x0,tol,Nmax):
    count = 0
    while (count <Nmax):
        count = count +1
        x1 = f(x0)
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            return [xstar,ier]
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]

def driver():
    print("Problem: [Root, Success]")
    print("1a: ",bisection(f1,.5,2,10** (-5)))
    print("1b: ",bisection(f1,-1,.5,10** (-5)))
    print("1c: ",bisection(f1,-1,2,10** (-5)))
    print("2a: ",bisection(f2,0,2.4,10** (-5)))
    print("2b: ",bisection(f3,0,2,10** (-5)))
    print("2ci: ",bisection(f4,0,.1,10** (-5)))
    print("2cii: ",bisection(f4,0.5,3*np.pi/4,10** (-5)))
   # print("3a: ", fixedpt(f5,1, 10**(-10), 50 ), "f(7^1/5): ", f5(7**(1.5)), "= 7^1/5")
    # print("3a: ", fixedpt(f6,1, 10**(-10), 50 ), "f(7^1/5): ", f6(7**(1.5)), "= 7^1/5")
    print("3c: ", fixedpt(f7,1, 10**(-10), 50 ), "f(7^1/5): ", f7( 7**(1.5) ), "= 7^1/5")
    print("3d: ", fixedpt(f8,1, 10**(-10), 50 ), "f(7^1/5): ", f8( 7**(1.5) ), "= 7^1/5")
driver()