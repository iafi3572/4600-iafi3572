import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import math
epsilon = np.finfo(float).eps
def driver(N,verb1 =True, verb2 = True,verb3 = True,verb4 = True):

    f = lambda x: 1./(1.+x**2)
    fp = lambda x: -2*x/(1.+x**2)**2
    fp0 = fp(-5)
    fpN = fp(5)
    ''' interval'''
    a = -5
    b = 5
   
    ''' create equispaced interpolation nodes'''
    xint = np.linspace(a,b,N+1)
    xint1 = np.array([
        0.5 * (b + a) + 0.5 * (b - a) * np.cos((N - i) * np.pi / N) 
        for i in range(N+1)
    ])    
    ''' create interpolation data'''
    yint = np.zeros(N+1)
    yint1 = np.zeros(N+1)
    ypint = np.zeros(N+1)
    ypint1 = np.zeros(N+1)

    for jj in range(N+1):
        yint[jj] = f(xint[jj])
        yint1[jj] = f(xint1[jj])
        ypint[jj] = fp(xint[jj])
        ypint1[jj] = fp(xint1[jj])
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    (M,C,D) = create_natural_spline(yint,xint,N)
    (M1,C1,D1) = create_natural_spline(yint1,xint1,N)
    (Ma,Ca,Da) = create_clamped_spline(yint,xint,N,fp0,fpN)
    (Ma1,Ca1,Da1) = create_clamped_spline(yint1,xint1,N,fp0,fpN)
    yevalL = np.zeros(Neval+1)
    yevalH = np.zeros(Neval+1)
    yevalC = eval_cubic_spline(xeval,Neval,xint,N,M,C,D)
    yevalD = eval_cubic_spline(xeval,Neval,xint,N,Ma,Ca,Da)
    yevalL1 = np.zeros(Neval+1)
    yevalH1 = np.zeros(Neval+1)
    yevalC1 = eval_cubic_spline(xeval,Neval,xint1,N,M1,C1,D1)
    yevalD1 = eval_cubic_spline(xeval,Neval,xint1,N,Ma1,Ca1,Da1)
    for kk in range(Neval+1):
      yevalL[kk] = eval_lagrange(xeval[kk],xint,yint,N)
      yevalH[kk] = eval_hermite(xeval[kk],xint,yint,ypint,N)
      yevalL1[kk] = eval_lagrange(xeval[kk],xint1,yint1,N)
      yevalH1[kk] = eval_hermite(xeval[kk],xint1,yint1,ypint1,N)

    ''' create vector with exact values'''
    fex = np.zeros(Neval+1)
    for kk in range(Neval+1):
        fex[kk] = f(xeval[kk])
    
    if verb1:
        plt.figure()
        plt.title("Lagrange and Hermite\nEquipsaced Nodes N = {N}".format(N=N))
        plt.plot(xeval,fex,label = "f(x)")
        plt.plot(xeval,yevalL,label='Lagrange') 
        plt.plot(xeval,yevalH,label='Hermite')
        plt.legend()
        plt.semilogy()
        plt.savefig("problem1.{N}LnH.png".format(N=N))
    if verb2:
        plt.figure()
        plt.title("Splines\nEquipsaced Nodes N = {N}".format(N=N))
        plt.plot(xeval,fex,label = "f(x)")
        plt.plot(xeval,yevalC,label='Natural Spline',color = "purple") 
        plt.plot(xeval,yevalD,label='Clamped Spline',color = "red") 
        plt.legend()
        plt.semilogy()
        plt.savefig("problem1.{N}spline.png".format(N=N))
    if verb3:
        plt.figure()
        plt.title("Lagrange and Hermite\nChebychev Nodes N = {N}".format(N=N))
        plt.plot(xeval,fex,label = "f(x)")
        plt.plot(xeval,yevalL1,label='Lagrange') 
        plt.plot(xeval,yevalH1,label='Hermite')
        plt.legend()
        plt.semilogy()
        plt.savefig("problem2.{N}LnH.png".format(N=N))
    if verb4:
        plt.figure()
        plt.title("Splines\nChebychev Nodes N = {N}".format(N=N))
        plt.plot(xeval,fex,label = "f(x)")
        plt.plot(xeval,yevalC1,label='Natural Spline',color = "purple") 
        plt.plot(xeval,yevalD1,label='Clamped Spline',color = "red") 
        plt.legend()
        plt.semilogy()
        plt.savefig("problem2.{N}spline.png".format(N=N))
    ''' 
    errL = abs(yevalL-fex)
    errH = abs(yevalH-fex)
    plt.figure()
    plt.semilogy(xeval,errL,'bs--',label='Lagrange')
    plt.semilogy(xeval,errH,'c.--',label='Hermite')
    plt.show()            
    '''

def eval_hermite(xeval,xint,yint,ypint,N):

    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])
              
              
    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
#    lpj2 = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
#              lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
              lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])

    yeval = 0.
     
    for jj in range(N+1):
       Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
       Rj = (xeval-xint[jj])*lj[jj]**2
#       if (jj == 0):
#         print(Qj)
         
#         print(Rj)
#         print(Qj)
#         print(xeval)
 #        return
       yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj
       
    return(yeval)
       


def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)
  
def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
        hi = xint[i]-xint[i-1]
        hip = xint[i+1] - xint[i]

        b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
        h[i-1] = hi
        h[i] = hip

#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0][0] = 1.0
    for j in range(1,N):
       A[j][j-1] = h[j-1]/6
       A[j][j] = (h[j]+h[j-1])/3 
       A[j][j+1] = h[j]/6
    A[N][N] = 1
    Ainv = inv(A)
    
    #M  = Ainv.dot(b)
    M = np.linalg.solve(A, b) 
#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = yint[j]/h[j]-h[j]*M[j]/6
       D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)

def create_clamped_spline(yint, xint, N, fp0, fpN):
    # Create the right-hand side for the linear system
    b = np.zeros(N + 1)
    h = np.zeros(N + 1)  # Step sizes
    
    for i in range(1, N):
        hi = xint[i] - xint[i-1]
        hip = xint[i+1] - xint[i]
        b[i] = (yint[i+1] - yint[i]) / hip - (yint[i] - yint[i-1]) / hi
        h[i-1] = hi
        h[i] = hip
    
    # added for clamped boundary conditions
    h[N] = xint[N] - xint[N-1]
    b[0] = (yint[1] - yint[0]) / h[0] - fp0
    b[N] = fpN - (yint[N] - yint[N-1]) / h[N-1]
    # Construct the matrix A for solving M values
    A = np.zeros((N + 1, N + 1))
    # changed for clamped condition
    A[0][0] = 2 / h[0]  
    A[0][1] = 1 / h[0]
    
    for j in range(1, N):
        A[j][j-1] = h[j-1] / 6
        A[j][j] = (h[j] + h[j-1]) / 3
        A[j][j+1] = h[j] / 6
    # changed for clamped
    A[N][N-1] = 1 / h[N-1]
    A[N][N] = 2 / h[N-1]  

    # Solve for M
    Ainv = inv(A)
    #M = Ainv.dot(b)
    M = np.linalg.solve(A, b) 
    # Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j] / h[j] - h[j] * M[j] / 6
        D[j] = yint[j+1] / h[j] - h[j] * M[j+1] / 6
    
    return M, C, D

def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i

    hi = xip-xi
    yeval = (Mi*(xip-xeval)**3 +(xeval-xi)**3*Mip)/(6*hi) \
            + C*(xip-xeval) + D*(xeval-xi)
    return yeval 
    
def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
#        print('yloc = ', yloc)
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)
     
driver(5)  
driver(10)  
driver(15)  
driver(20)  
         

   
   
