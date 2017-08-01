from scipy import linalg as la
import numpy as np
from matplotlib import pyplot as plt
import cmath
import math
from matplotlib import pyplot as plt

def steep_de(x0, Q, b, Ftol, Fdist, Fmaxiter):
    Fiter = 1
    while Fdist > Ftol and Fiter < Fmaxiter:
        Df = np.dot(Q, x0) - b
        alphak = np.dot(Df, Df.T)/np.dot(np.dot(Df, Q), Df.T)
        x1 = x0 - alphak * Df
        Fdist = la.norm(x1-x0)
        x0 = x1
        Fiter += 1
        
    if Fiter < Fmaxiter:
        print('function converged after this many iterations:', Fiter)
    else:
        print('function did not converge')
    return x1

Ftol = 1e-6
Fdist = 7.0
Fmaxiter = 3000
Q = np.array([[-1, 2], [2, 3]])
b = np.array([1, 1])
x0 =x_init = np.array([1, 1])

print(steep_de(x_init, Q, b, Ftol, Fdist, Fmaxiter))

