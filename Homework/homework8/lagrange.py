import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():
    f = lambda x: 1/(1+x**2)

    N = [5,10,15,20]
    ''' interval'''
    a = -5
    b = 5
   
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''

    ''' evaluate lagrange poly '''
    for i in range(4):
        xint = np.linspace(a,b,N[i]+1)
        yint = f(xint)
        for kk in range(Neval+1):
            yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N[i])
            
        ''' create vector with exact values'''
        fex = f(xeval)
        

        plt.figure()    
        plt.plot(xeval,fex,label = "f(x)")
        plt.plot(xeval,yeval_l, label = "Lagrange") 
        plt.savefig("problem1.{i}.png".format(i=i))
        '''
        plt.figure() 
        err_l = abs(yeval_l-fex)
        plt.semilogy(xeval,err_l,label='lagrange')
        plt.legend()
        plt.show()
        '''
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
