import numpy as np
import matplotlib.pyplot as plt


def partB():
    x1= np.pi
    x2= 10**6
    a = np.arange(16,-1,-1)
    d = 1/(10**a)
    fx1 = abs(-2*np.sin((2*x1 +d)/2)*(np.sin(d/2)) - (np.cos(x1+d) - np.cos(x1)))
    fx2 = abs(-2*np.sin((2*x2 +d)/2)*(np.sin(d/2)) - (np.cos(x2+d) - np.cos(x2)))
    plt.plot(np.log10(d),fx1, label= r'$x = \pi$')
    plt.plot(np.log10(d),fx2, label= r'$x= 10^6$')
    plt.xlabel(r'$\log_{10}(\delta)$')
    plt.ylabel(r'$f(x)$')
    plt.title(r'Absoulute Difference Between $sin(\frac{2x+\delta}{2})sin(\frac{\delta}{2})$ and $cos(x+\delta) - cos(x)  $')
    plt.legend()
    plt.savefig("Problem5.png")
    
partB()