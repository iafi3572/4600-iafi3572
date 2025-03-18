import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad


def eval_legendre(n,x):
    p = np.zeros(n+1)
    p0= 1
    p1=1
    p[0] = p0
    if(n==0):
       return p
    p[1] = p1
    if (n==1):
       return p
    for i in range(2,n+1):
        pn = (((2*i+1) * x * p1) - i*p0)/(i+1)
        p[i] = pn
        p0 = p1
        p1 = pn

    return p


def driver():

#  function you want to approximate
    f = lambda x: 1/(1+x**2)

# Interval of interest    
    a = -1
    b = 1
# weight function    
    w = lambda x: 1.

# order of approximation
    n = 2

#  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)

    for kk in range(N+1):
      pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
      
    ''' create vector with exact values'''
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
        
    plt.figure()    
    plt.plot(xeval,fex,'ro-', label= 'f(x)')
    plt.plot(xeval,pval,'bs--',label= 'Expansion') 
    plt.legend()
    plt.show()    
    
    err_l = abs(pval-fex)
    plt.semilogy(xeval,err_l,'ro--',label='error')
    plt.legend()
    plt.show()
       

def eval_legendre_expansion(f, a, b, w, n, x):
    """Evaluate the Legendre expansion."""
    # Initialize Legendre polynomials and coefficients
    pval = 0.0
    for j in range(n + 1):
        # Define the Legendre polynomial phi_j(x)
        phi_j = lambda x: eval_legendre(j, x)[j]
        
        # Define phi_j^2(x) * w(x)
        phi_j_sq = lambda x: w(x) * phi_j(x)**2
        
        # Normalize the polynomials using quad (integrate over [a, b])
        norm_fac, _ = quad(phi_j_sq, a, b)
        
        # Define the function for the coefficient calculation
        func_j = lambda x: (phi_j(x) * f(x) * w(x)) / norm_fac
        
        # Calculate the coefficient using quad
        aj, _ = quad(func_j, a, b)
        
        # Add the contribution of this term to the Legendre expansion
        pval += aj * phi_j(x)

    return pval
    
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()               
