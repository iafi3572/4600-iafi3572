import numpy as np

#both converge in 7 iterations
def forward(f,h,x):
    return (f(x+h)-f(x))/h
def centered(f,h,x):
    return (f(x+h) - f(x-h))/(2*h)
def driver():
    x = np.pi/2
    f = lambda x: np.cos(x)
    h = .01*2.**(-np.arange(0,10))
    print(forward(f,h,x))
    print(centered(f,h,x))
driver()
