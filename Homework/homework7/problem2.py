import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():
    f = lambda x: 1/(1+(10*x)**2)
    N = 19
    a = -1
    b = 1
    ''' create equispaced interpolation nodes'''
    i = np.array(range(1,N+1))
    xint = -1+ ((i-1)* 2 )/ (N-1)

#    print('xint =',xint)
    '''Create interpolation data'''
    yint = f(xint)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''

    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N-1)
        


    ''' create vector with exact values'''
    fex = f(xeval)
       

    plt.figure()    
    plt.plot(xeval,fex,label = "f(x)")
    plt.plot(xeval,yeval_l, label = "Lagrange Interpolation ") 
    #err_l = abs(yeval_l-fex)
  #  plt.plot(xeval,err_l, label = "Error")
    plt.legend()
    plt.savefig("problem2.png")
    # plt.show()
    

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


driver()        
