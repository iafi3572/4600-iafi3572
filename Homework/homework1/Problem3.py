import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return abs((np.sin(x)-17*x*np.sin(x)+3*np.cos(x)-9*(x**2)*np.cos(x)+(x**3)*np.sin(x)))/48
xvals = np.linspace(-1,1,100)
plt.plot(xvals,f(xvals))
plt.savefig('Problem3')
print(max(f(xvals)))

def error():
    return abs(np.cos(.5)*(1-1/2+1/8) -1 - 1/2 +1/8)
print(error())