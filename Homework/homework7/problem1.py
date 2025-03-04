import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt 

def driver(): 
    f = lambda x: 1/(1+(10*x)**2)
    N = 20
    a = -1
    b = 1
    
    ''' Create interpolation nodes'''
    i = np.array(range(1,N+1))
    xint = -1+ ((i-1)* 2 )/ (N-1)

#    print('xint =',xint)
    '''Create interpolation data'''
    yint = f(xint)
#    print('yint =',yint)
    
    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint,N)
#    print('V = ',V)

    ''' Invert the Vandermonde matrix'''    
    Vinv = inv(V)
    
#    print('Vinv = ' , Vinv)
    
    ''' Apply inverse to rhs'''
    ''' to create the coefficients'''
    coef = Vinv @ yint
    x = np.linspace(a,b,1001)
    
    yeval = eval_monomial(x,coef,N)
    plt.plot(xint,yint,'o')
    plt.plot(x,f(x))
    plt.plot(x,yeval)
    plt.savefig("problem1.png")
    print(max(yeval))


def eval_monomial(xeval, coef, N):
    yeval = np.zeros_like(xeval)
    for j in range(N):
        yeval += coef[j] * xeval**j
    return yeval


def Vandermonde(xint, N):
    V = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            V[j][i] = xint[j]**i
    return V     


driver()    
