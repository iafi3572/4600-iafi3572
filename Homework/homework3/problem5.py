import matplotlib.pyplot as plt
import numpy as np

def fixedpt(f,x0,tol,Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    count = 0
    g = lambda x: -np.sin(2*x) +5*x/4-3/4
    while (count <Nmax):
        count = count +1
        x1 = g(x0)
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            return [xstar,ier]
        x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier]


def driver():
    f = lambda x: x -4*np.sin(2*x) -3
    x = np.linspace(-5,5,100)
    plt.plot(x,f(x),)
    plt.axvline(0,c='black')
    plt.axhline(0,c='black')
    plt.xlim(-5,5)
    plt.ylim(-10,10)
    plt.ylabel(r'$f(x) = x - 4\sin(2x) -3$')
    plt.xlabel(r'$x$')
    plt.savefig("problem5.png")
    guesses = [-1,0,1.5,3,5]
    for x0 in guesses:
        [xstar,ier] = (fixedpt(f,x0,10**-10,100))
        print(f"Guess: {x0}, Root: {xstar:.10f}")
driver()

