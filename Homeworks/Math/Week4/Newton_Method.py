from scipy import linalg as la
from numba import jit

def f(x):
	return x**4 - 3
	
def f_prime(x):
	return 4*x**3

def f_2_prime(x):
	return 12*x**2

N = 10000
x0 = 1

for i in range(N):
    x1 = x0 - (f_prime(x0)/f_2_prime(x0))
    if abs(x1 - x0) < x0 * tol:
        print("The Value =", x1)
        print("end")
        break
    else:
        x0 = x1



       
       
       
       
       
       
       
       