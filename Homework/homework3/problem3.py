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
            return [astar, ier,count]
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
    f = lambda x : x**3 +x -4
    a = 1
    b =4 
    tol = 10**(-3)
    print(bisection(f,a,b,tol))
driver()