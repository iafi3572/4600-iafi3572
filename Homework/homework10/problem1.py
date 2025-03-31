import matplotlib.pyplot as plt
import numpy as np

def driver():
    x = np.linspace(0,5)
    f = lambda x: np.sin(x)
    a = lambda x: (x-(7*x**3)/60)/(1+x**2/20)
    b = lambda x: x/(1+x**2/6 + (7*x**4)/360)
    t = lambda x: x - x**3/6 + x**5 /120
    plt.plot(x,a(x),label = "(3,3) and (4,2)")
    plt.plot(x,b(x),label = "(2,4)")
    plt.plot(x,t(x), label = "Maclaurin")
    plt.plot(x,f(x),label = "sin(x)")
    plt.legend()
    plt.savefig("problem1.png")
driver()