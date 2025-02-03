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
def driver():
    f1 = lambda x: (x-5)**9
    f2 = lambda x: x**9-45*x**8+900*x**7-10500*x**6+78750*x**5-393750*x**4+1312500*x**3-2812500*x**2+3515625*x-1953125
    a = 4.82
    b= 5.2
    tol = 10**(-4)
    print(bisection(f1,a,b,tol))
    print(bisection(f2,a,b,tol))
driver()