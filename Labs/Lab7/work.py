import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
def monomial(V, y):
   coefs = la.solve(V,y)   
   return coefs
def driver():
   f = lambda x: 1/(1+(10*x)**2)
   y = np.array([f(-1),f(1)])
   V = np.array([[1, -1],[1,1]])
   print(monomial(V,y))
driver()