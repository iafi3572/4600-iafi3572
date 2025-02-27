import numpy as np
from numpy import random as rand
import time
import math
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
from numpy.linalg import norm
def F(x):
    return np.array([x[0] + np.cos(x[0] * x[1] *x[2]) , (1-x[0])**(1/4) + x[1] + .05*x[2]**2 - 0.15 * x[2] -1,-x[0]**2 -.1*x[1]**2 + .01*x[1] + x[2] -1])
def JF(x):
    J11 = 1 -np.sin(x[0]*x[1]*x[2]) * x[1] * x[2]
    J12 =  -np.sin(x[0]*x[1]*x[2]) * x[0] * x[2]
    J13 =  -np.sin(x[0]*x[1]*x[2]) * x[0] * x[1]
    J21 = (1/4) *(1-x[0])**(-3/4)
    J22 = 1
    J23 = .1 * x[2] - .15
    J31 = -2*x[0]
    J32 = -.2 * x[1] + .01
    J33 = 1
    return np.array([[ J11,J12,J13 ],[J21 ,J22,J23 ],[J31,J32,J33]])
def evalg(x):
    W = F(x)
    g = W[0]**2 + W[1]**2 + W[2]**2
    return g
def eval_gradg(x):

    W = F(x)
    J = JF(x)
    gradg = np.transpose(J).dot(W)
    return gradg
    
def driver():
    ############################################################################
    ############################################################################
    # Rootfinding example start. You are given F(x)=0.

    #First, we define F(x) and its Jacobian.

     #Apply Newton Method:
    x0 = np.array([-1,1,3]); tol=1e-14; nmax=100;
    (rN,rnN,nfN,nJN) = newton_method_nd(F,JF,x0,tol,nmax,True);
    print(rN)
    [x,g1,ier,its] = SteepestDescent(x0,tol,nmax)
    print(its)
################################################################################
# Newton method in n dimensions implementation
def newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;

    if (len(x0)<100):
        if (np.linalg.cond(Jf(x0)) > 1e16):
            print("Error: matrix too close to singular");
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
            r=x0;
            return (r,rn,nf,nJ);

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        # compute n x n Jacobian matrix
        Jn = Jf(xn);
        nJ+=1;

        if verb:
            print("|--%d--|%1.7f|%1.15f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

def SteepestDescent(x,tol,Nmax):
    
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier,its]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,ier,its]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier,its]

# Execute driver
driver()
