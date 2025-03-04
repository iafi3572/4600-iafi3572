def driver():
    f = lambda x: x**2
    x0 =0
    x1 = 2
    alpha = 1
    print(subroutine(x0,x1,f,alpha))

def subroutine(x0,x1,f,alpha):
    y0 = f(x0)
    y1 = f(x1)
    m = (y1-y0)/(x1-x0)
    return m*(alpha-x0) + y0

driver()