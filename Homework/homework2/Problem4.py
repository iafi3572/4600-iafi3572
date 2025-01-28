import numpy as np
import random
import matplotlib.pyplot as plt


def summ():
    t = np.arange(0,np.pi+.001,np.pi/30)
    y = np.cos(t)
    total =0
    for i in range(t.size):
        total += t[i] * y[i]
    print("the sum is: ", total)

def graph():
    #part a
    theta = np.linspace(0,2*np.pi)
    R = 1.2
    d = .1
    f = 15
    p =0
    x = lambda t: R*(1+d*np.sin(f*t+p)) * np.cos(t)
    y = lambda t: R*(1+d*np.sin(f*t+p)) * np.sin(t)
    plt.figure(1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title("Problem 4b First Plot")
    plt.plot(x(theta),y(theta))
    plt.savefig("Problem4bi.png")
    # part b
    plt.figure(2)
    for i in range(1,11):
        R = i
        d = .1
        f = 2+i
        p = random.uniform(0,2)
        x = lambda t: R*(1+d*np.sin(f*t+p)) * np.cos(t)
        y = lambda t: R*(1+d*np.sin(f*t+p)) * np.sin(t)
        plt.plot(x(theta),y(theta))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title("Problem 4b Second Plot")
    plt.savefig("Problem4bii.png")

graph()
summ()