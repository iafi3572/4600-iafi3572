import numpy as np
from numpy.linalg import inv 
import matplotlib.pyplot as plt

def driver():
    f = lambda x: 1/(1+x**2)
    fp = lambda x: -1/(1+x**2)**2 * 2*x
    a = -5
    b = 5
    num = 100
    ''' generating nodes'''
    N = [5,10,15,20]
    xeval = np.linspace(-5,5,100)
    xint = [np.linspace(a,b,j) for j in N]
    yint = [ f(xint[j]) for j in range(4)]
    ypint = [ fp(xint[j]) for j in range(4)]
    xintc = [np.cos(((2*np.arange(1, (i)+1) - 1) * np.pi) / (2 * (i))) for i in N]
    yintc = [ f(xintc[j]) for j in range(4)]
    ypintc = [ fp(xintc[j]) for j in range(4)]

    # -----------------------------------------------------
    for i in range(4):
        # note that this maybe graphs them all on top of eachother
        Leval = np.zeros(num)
        Heval = np.zeros(num)
        for j in range(num):
            Leval[j] = lagrange(xeval[j],xint[i],yint[i],N[i])
            Heval[j] = hermite(xeval[j],xint[i],yint[i],ypint[i],N[i])

        (M,C,D) = create_natural_spline(yint[i],xint[i],N[i])
        nCeval = eval_cubic_spline(xeval,num,xint[i],N[i],M,C,D)

      #  cCeval = 0 # TODO
        plt.figure(1)
        plt.title(f'Equispaced Nodes\nn = {N[i]}')
        plt.plot(xeval,f(xeval))
        plt.plot(xint[i],yint[i],'o')
        plt.plot(xeval,Leval,label = "Lagrange")
        plt.plot(xeval,Heval, label = "Hermite")
        plt.plot(xeval,nCeval, label = "Natural Cubic spline")
      #  plt.plot(xeval,cCeval, label = "Clamped Cubic spline")
        plt.legend()
        plt.savefig("problem1.{i}.png".format(i=i))
        
        Levalc = lagrange(xeval,xintc[i],yintc[i],N[i])
        Hevalc = hermite(xeval,xintc[i],yintc[i],ypintc[i],N[i])
        nCevalc = 0 # TODO
     #   cCevalc = 0 # TODO
        plt.figure(2)
        plt.title(f'Chebychev Nodes\nn = {N[i]}')
        plt.plot(xintc[i],yintc[i],'o')
        plt.plot(xeval,f(xeval))
        plt.plot(xeval,Levalc,label = "Lagrange")
        plt.plot(xeval,Hevalc, label = "Hermite")
        plt.plot(xeval,nCevalc, label = "Natural Cubic spline")
     #   plt.plot(xeval,cCevalc, label = "Clamped Cubic spline")
        plt.savefig("problem2.{i}.png".format(i=i))

    
def lagrange(xeval,xint,yint,N):
    lj = np.ones(N+1)
    for count in range(N):
       for jj in range(N):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)

def hermite(xeval,xint,yint,ypint,N):
    
    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N+1)
    for count in range(N):
       for jj in range(N):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
#    lpj2 = np.ones(N+1)
    for count in range(N):
       for jj in range(N):
           if (jj != count):
#              lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
              lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
              

    yeval = 0.
    
    for jj in range(N):
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
       

def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N-1):
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
    
    M  = Ainv.dot(b)

#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = yint[j]/h[j]-h[j]*M[j]/6
       D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)
       
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
                      



def clampedCubic():
    return 0
driver()