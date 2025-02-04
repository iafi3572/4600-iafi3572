import numpy as np


def approx(p):
    pt = np.zeros(np.size(p),1)
    for i in (range(np.size(p))-2):
        pt[i] = p[i] - ((p[i+1] - p[i])**2)/(p[i+2]-2*p[i+1] + p[i])
    return pt
def aitkens():
    # please madi I tried idk what's going on
