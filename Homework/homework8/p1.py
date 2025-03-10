import numpy as np
import matplotlib.pyplot as plt

def driver():
    f = lambda x: 1/(1+x**2)
    fp = lambda x: -1/(1+x**2)**2 * 2*x
    a = -5
    b = 5
    ''' generating nodes'''
    N = [5,10,15,20]
    xeval = np.linspace(-5,5)
    xint = [np.linspace(a,b,j) for j in N]
    yint = [ f(xint[j]) for j in range(4)]
    ypint = [ fp(xint[j]) for j in range(4)]
    xintc = [np.cos(((2*np.arange(1, (i)+1) - 1) * np.pi) / (2 * (i))) for i in N]
    yintc = [ f(xintc[j]) for j in range(4)]
    ypintc = [ fp(xintc[j]) for j in range(4)]

    # -----------------------------------------------------
    for i in range(4):
        # note that this maybe graphs them all on top of eachother
        Leval = lagrange(xeval,xint[i],yint[i],N[i])
        Heval = hermite(xeval,xint[i],yint[i],ypint[i],N[i])
        nCeval = 0 # TODO
        cCeval = 0 # TODO
        plt.figure(1)
        plt.title(f'Equispaced Nodes\nn = {N[i]}')
        plt.plot(xeval,f(xeval))
        plt.plot(xint[i],yint[i],'o')
        plt.plot(xeval,Leval,label = "Lagrange")
        plt.plot(xeval,Heval, label = "Hermite")
        plt.plot(xeval,nCeval, label = "Natural Cubic spline")
        plt.plot(xeval,cCeval, label = "Clamped Cubic spline")
        plt.legend()
        plt.savefig("problem1.{i}.png".format(i=i))
        
        Levalc = lagrange(xeval,xintc[i],yintc[i],N[i])
        Hevalc = hermite(xeval,xintc[i],yintc[i],ypintc[i],N[i])
        nCevalc = 0 # TODO
        cCevalc = 0 # TODO
        plt.figure(2)
        plt.title(f'Chebychev Nodes\nn = {N[i]}')
        plt.plot(xintc[i],yintc[i],'o')
        plt.plot(xeval,f(xeval))
        plt.plot(xeval,Levalc,label = "Lagrange")
        plt.plot(xeval,Hevalc, label = "Hermite")
        plt.plot(xeval,nCevalc, label = "Natural Cubic spline")
        plt.plot(xeval,cCevalc, label = "Clamped Cubic spline")
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
       

def natrualCubic():
    return 0

def clampedCubic():
    return 0
driver()