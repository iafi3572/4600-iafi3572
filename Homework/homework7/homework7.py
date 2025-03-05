import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt 

def driver(): 
    f = lambda x: 1/(1+(10*x)**2)
    N = 21
    a = -1
    b = 1
    
    ''' Create interpolation nodes'''
    i = np.array(range(1,N+1))
    xint = -1+ ((i-1)* 2 )/ (N-1)
    x3int = np.cos( ((2*i-1)*np.pi)/(2*N) )
#    print('xint =',xint)
    '''Create interpolation data'''
    yint = f(xint)
    y3int = f(x3int)
#    print('yint =',yint)
    
    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint,N)
    V3 = Vandermonde(x3int,N)
#    print('V = ',V)

    ''' Invert the Vandermonde matrix'''    
    Vinv = inv(V)
    V3inv = inv(V3)
    
#    print('Vinv = ' , Vinv)
    
    ''' Apply inverse to rhs'''
    ''' to create the coefficients'''
    coef = Vinv @ yint
    
    coef3 = V3inv @ y3int

    x = np.linspace(a,b,1001)
    p = evalP(x,xint,yint)
    yeval = eval_monomial(x,coef,N)
    y3eval = eval_monomial(x,coef3,N)
    plt.figure()
    plt.ylim(-20,50)
    plt.plot(xint,yint,'o')
    plt.plot(x,f(x), label = "f(x)")
    plt.plot(x,yeval,label = "Interpolation")
    plt.legend()
    plt.savefig("problem1.png")

    plt.figure()
    plt.plot(x3int,y3int,'o')
    plt.plot(x,f(x), label = "f(x)")
    plt.plot(x,y3eval,label = "Interpolation")
    plt.legend()
    plt.savefig("problem3.png")

    plt.figure()
    plt.ylim(-20,45)
    plt.plot(xint,yint,'o')
    plt.plot(x,f(x), label = "f(x)")
    plt.plot(x,p,label = "Interpolation")
    plt.legend()
    plt.savefig("problem2.png")



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

def evalP(x, xint, yint):
    phi = np.ones_like(x)
    for i in range(xint.size):
        phi *= (x - xint[i])

    sum = np.zeros_like(x)
    for j in range(xint.size):
        w = 1
        for i in range(xint.size):
            if i != j:
                w *= 1 / (xint[j] - xint[i])
        
        numerator = np.where(x == xint[j], yint[j], w * yint[j])
        denominator = np.where(x == xint[j], 1, x - xint[j])
        sum += numerator / denominator

    return phi * sum


driver()    
