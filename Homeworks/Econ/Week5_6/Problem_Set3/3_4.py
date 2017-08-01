import numpy as np
import sympy as sy
import scipy.optimize as opt
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy import Function
from matplotlib import pyplot as plt

# Parameters
alpha = 0.35
beta = 0.98
rho = 0.95
sigma = 0.02
zbar = 0

A = alpha * beta
kbar = A**(1/(1-alpha))

k_grid = np.linspace(0.001, 5*kbar, 1000)

def true_closed(kbar, k_grid):
    optK = A * np.exp(zbar) * k_grid**alpha
    return optK

def fisrt_deri(kbar, k_grid):
    u = sy.symbols('u')
    x = sy.symbols('x')
    y = sy.symbols('y')

    ''' 
    This is wrong way, but keep for the future.
    
    F = lambda u, x, y: 1 / (u**alpha - x(u)) - beta*(alpha*x(u)**(alpha - 1)) / (x(u)**alpha - y(x(u)))
    Fy = lambdify((u, x(u), y(x(u))), sy.diff(F(u, x, y), y(x(u))))
    Fx = lambdify((u, x(u), y(x(u))), sy.diff(F(u, x, y), x(u)))
    Fu = lambdify((u, x(u), y(x(u))), sy.diff(F(u, x, y), u))

    a = Fy(kbar, kbar, kbar)
    b = Fx(kbar, kbar, kbar)
    c = Fu(kbar, kbar, kbar)
    '''
    
    F = lambda u, x, y: 1 / (u**alpha - x) - beta*(alpha*x**(alpha - 1)) / (x**alpha - y)
    Fy = lambdify((u, x, y), sy.diff(F(u, x, y), y))
    Fx = lambdify((u, x, y), sy.diff(F(u, x, y), x))
    Fu = lambdify((u, x, y), sy.diff(F(u, x, y), u))
    
    a = Fy(kbar, kbar, kbar)
    b = Fx(kbar, kbar, kbar)
    c = Fu(kbar, kbar, kbar)
    
    xu_plus = ( -b + np.sqrt(b**2 - 4*a*c) ) / (2*a)
    xu_minus = ( -b - np.sqrt(b**2 - 4*a*c) ) / (2*a)
    
    first_ap = kbar + xu_plus * (k_grid - kbar)
    return first_ap

def second_deri(kbar, k_grid):
    u = sy.symbols('u')
    x = sy.symbols('x')
    y = sy.symbols('y')

    F = lambda u, x, y: 1 / (u**alpha - x) - beta*(alpha*x**(alpha - 1)) / (x**alpha - y)
    Fy = lambdify((u, x, y), sy.diff(F(u, x, y), y))
    Fx = lambdify((u, x, y), sy.diff(F(u, x, y), x))
    Fu = lambdify((u, x, y), sy.diff(F(u, x, y), u))
    
    Fyy = lambdify((u, x, y), sy.diff(Fy(u, x, y), y))
    Fyx = lambdify((u, x, y), sy.diff(Fy(u, x, y), x))
    Fyu = lambdify((u, x, y), sy.diff(Fy(u, x, y), u))
    Fxx = lambdify((u, x, y), sy.diff(Fx(u, x, y), x))
    Fxu = lambdify((u, x, y), sy.diff(Fx(u, x, y), u))
    Fuu = lambdify((u, x, y), sy.diff(Fu(u, x, y), u))
    
    # assign values into functions
    Fy = Fy(kbar, kbar, kbar)
    Fx = Fx(kbar, kbar, kbar)
    Fu = Fu(kbar, kbar, kbar)
    Fyy = Fyy(kbar, kbar, kbar)
    Fyx = Fyx(kbar, kbar, kbar)
    Fyu = Fyu(kbar, kbar, kbar)
    Fxx = Fxx(kbar, kbar, kbar)
    Fxu = Fxu(kbar, kbar, kbar)
    Fuu = Fuu(kbar, kbar, kbar)
    
    # first deri
    a = Fy
    b = Fx
    c = Fu
    
    xu_plus = ( -b + np.sqrt(b**2 - 4*a*c) ) / (2*a)
    xu_minus = ( -b - np.sqrt(b**2 - 4*a*c) ) / (2*a)

    # second deri
    xu = xu_plus
    xuu = -(Fyy*xu**4 +2*Fyx*xu**3 + 2*Fyu*xu**2 + Fxx*xu**2 + 2*Fxu*xu + Fuu) / (Fy*xu**2 + Fy*xu + Fx)
    
    second_ap = kbar + xu * (k_grid - kbar) + 0.5*xuu * (k_grid - kbar)**2
    return second_ap

plt.plot(k_grid, true_closed(kbar, k_grid), label = "true")
plt.plot(k_grid, fisrt_deri(kbar, k_grid), label = "first")
plt.plot(k_grid, second_deri(kbar, k_grid), label = "second")
plt.legend()
plt.show()




























