import numpy as np
import sympy as sy
import scipy.optimize as opt
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy import Function
from matplotlib import pyplot as plt

alpha = 0.33
k = 5
z = 1
b = 2
t = 0.1
h = 24

fixed_args = (alpha, k, z, b, t, h)
def finding_w(w, *args):
    alpha, k, z, b, t, h = args
    nd = ( ((1-alpha)*z)/w )**(1/alpha) * k
    pi = z*k**alpha*(nd)**(1-alpha) - w*nd
    ns = h - (b/(w*(1+b)))*(w*h+pi - t)
    error = nd - ns
    return error

w_init = 0.2
w = opt.fsolve(finding_w, w_init, args = fixed_args)
print(w)

def finding_w_first_second(w_init, k0, k, h =1e-5):
    f = lambda k0: function(k0, w_init, fixed_args)
    first = (f(k0 + h) - f(k0 - h)) / (2*h)
    second = ( f(k0 + h) - 2* f(k0) + f(k0 - h) ) / (h**2)
    
    first_ap = f(k0) + first*(k - k0)
    second_ap = f(k0) + first*(k - k0) + 0.5*second*(k - k0)**2
    true_val = f(k)
    return true_val, first_ap, second_ap

def function(k0, w_init, *args):
    fixed_args = (alpha, k0, z, b, t, h)
    return opt.fsolve(finding_w, w_init, args = fixed_args)[0]
    
true_val, first_ap, second_ap = finding_w_first_second(w_init, 5, 5.001, h =1e-5)

print(first_ap, second_ap)    
 
k_grid = np.linspace(1, 15, 100)

tru = []
fir = []
sec = []
for k in k_grid:
    temp0, temp1, temp2 = finding_w_first_second(w_init, 5, k, h =1e-5)
    tru.append(temp0)
    fir.append(temp1)
    sec.append(temp2)

plt.plot(tru, label = "true")
plt.plot(fir, label = "first approximation")
plt.plot(sec, label = "second approximation")
plt.legend()
plt.show()

tru = []
fir = []
sec = []
for k in k_grid:
    temp0, temp1, temp2 = finding_w_first_second(w_init, 10, k, h =1e-5)
    tru.append(temp0)
    fir.append(temp1)
    sec.append(temp2)

plt.plot(tru, label = "true")
plt.plot(fir, label = "first approximation")
plt.plot(sec, label = "second approximation")
plt.legend()
plt.show()
  
    
    
    
    
    
    
    
    
    
    
    
    
    