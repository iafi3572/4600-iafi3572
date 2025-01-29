import math
import numpy as np
def alg(x):
    y = (math.e) **x
    return y -1
print(alg(0))
print(alg(9.999999995000000 * 10**(-10)))
print(np.expm1(9.999999995000000 * 10**(-10)))