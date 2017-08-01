import numpy as np
import sympy as sy
import scipy.optimize as opt
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy import Function
from matplotlib import pyplot as plt

x = 100
t = 0
y_init = 47
fixed_args = (x, t)

def finding_y(y, *args):
    x, t = args
    error = (x**0.35 + 0.9*x - y) ** -2.5 - 0.95 * (y**0.35 + 0.9 * y) ** -2.5
    return error

def finding_y_first_second_third(y_init, x0, x, h =1e-5):
    f = lambda x0: function(x0, y_init, fixed_args)
    first = (f(x0 + h) - f(x0 - h)) / (2*h)
    second = ( f(x0 + h) - 2* f(x0) + f(x0 - h) ) / (h**2)
    third = ( f(x0 + 2*h) - 2* f(x0 + h) + 2* f(x0 - h) - f(x0 - 2*h) ) / (2*h**3)
    
    first_ap = f(x0) + first*(x - x0)
    second_ap = f(x0) + first*(x - x0) + 0.5*second*(x - x0)**2
    third_ap = f(x0) + first*(x - x0) + 0.5*second*(x - x0)**2 + (1/6)*third*(x - x0)**3
    true_val = f(x)
    return true_val, first_ap, second_ap, third_ap

def function(x0, y_init, *args):
    fixed_args = (x0, t)
    return opt.fsolve(finding_y, y_init, args = fixed_args)[0]
    
true_val, first_ap, second_ap, third_ap = finding_y_first_second_third(y_init, 100, 100.2, h =1e-5)

x_grid = np.linspace(99, 101, 50)

tru = []
fir = []
sec = []
thir = []
for x in x_grid:
    temp0, temp1, temp2, temp3 = finding_y_first_second_third(y_init, 100, x, h =1e-2)
    tru.append(temp0)
    fir.append(temp1)
    sec.append(temp2)
    thir.append(temp3)

plt.plot(np.array(tru) - np.array(tru), label = "True")
plt.plot(np.array(fir) - np.array(tru), label = "First approximation")

plt.plot(np.array(sec) - np.array(tru), label = "Second approximation")
plt.plot(np.array(thir) - np.array(tru), label = "Third approximation")
plt.legend()
plt.show()


plt.plot(np.array(tru), label = "True")
plt.plot(np.array(fir), label = "First approximation")
plt.plot(np.array(sec), label = "Second approximation")
plt.plot(np.array(thir), label = "Third approximation")
plt.legend()
plt.show()

    