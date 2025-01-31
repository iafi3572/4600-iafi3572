import math
import numpy as np
def alg(x):
    y = (math.e) **x
    return y -1
print(alg(0))
print(alg(9.999999995000000 * 10**(-10)))
print(np.expm1(9.999999995000000 * 10**(-10)))

def partd():
    x = 9.999999995000000 * 10 **(-10)
    E = lambda n : (np.e**x * (x)**(n+1))/ abs(np.e**x -1)*(math.factorial(n+1))
    n =1
    while E(n) > 10**(-16):
        n = n+1
    return n
print(partd())

def parte():
    x = 9.999999995000000 * 10 **(-10)
    f = 10**-9
    aprx = lambda x: x+x**2/2
    return abs( f - aprx(x) )
print(parte())
