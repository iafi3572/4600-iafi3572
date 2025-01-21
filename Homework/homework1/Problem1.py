import numpy as np
from matplotlib import pyplot as plt

x = np.arange(1.920, 2.080, .001)
p1 = (x-2)**9
p2= x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 +5376*x**3 - 4608*x**2 +2304*x - 512
plt.plot(x,p1,label= '(x-2)^9')
plt.plot(x,p2, label = 'expanded')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title("Problem 1")
plt.legend()
plt.savefig('Problem1.png')

""" The non expanded version is smooth and looks how we would expect. The expanded version is all jagged
and is clearly not an accurate representation of the intended graph. This is likely due to the additional 
subtractions that take place in the expanded formula. This causes a higher loss of precision"""
