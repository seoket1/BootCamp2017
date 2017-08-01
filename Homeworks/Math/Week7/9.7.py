from scipy import linalg as la
import numpy as np
from matplotlib import pyplot as plt
import cmath
import math
from matplotlib import pyplot as plt

def f_forward_grad(F, X, Rerr):
    h = 2 * np.sqrt(Rerr)
    foward1 = []
    
    for i in range(len(X)):
        foward1.append( ( F(X+ h*np.eye(len(X))[i,:]) - F(X) ) / h )

    return np.array(foward1)


F = lambda x : x[0]**2 + x[1]**3
X = np.array([2, 3])
print(f_forward_grad(F, X, Rerr = 0.00001))

















