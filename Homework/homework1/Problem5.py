import numpy as np
import matplotlib.pyplot as plt
    
def graph(x1,x2):
    a = np.arange(16,-1,-1)
    d = 1/(10**a)
    fx1 = abs(-2*np.sin((2*x1 +d)/2)*(np.sin(d/2)) - (np.cos(x1+d) - np.cos(x1)))
    fx2 = abs(-2*np.sin((2*x2 +d)/2)*(np.sin(d/2)) - (np.cos(x2+d) - np.cos(x2)))
    fx3 = abs(-d*np.sin(x1) - (np.cos(x1+d) - np.cos(x1)))
    fx4 = abs(-d*np.sin(x2)- (np.cos(x2+d) - np.cos(x2)))
    plt.figure(1)
    plt.plot(np.log10(d),fx1, label= r'$x = \pi$')
    plt.plot(np.log10(d),fx2, label= r'$x= 10^6$'"\n")
    plt.xlabel(r'$\log_{10}(\delta)$')
    plt.ylabel(r'$f(x)$')
    plt.title("Absoulute Difference Part B")
    plt.legend()
    plt.savefig("Problem5.png")
    plt.figure(2)
    plt.plot(np.log10(d),fx3, label=r'$x = \pi$')
    plt.plot(np.log10(d),fx4, label= r'$x= 10^6$')
    plt.xlabel(r'$\log_{10}(\delta)$')
    plt.ylabel(r'$f(x)$')
    plt.title("Absoulute Difference Part C")
    plt.legend()
    plt.savefig("Problem5.2.png")
    
graph(np.pi,10**6)

