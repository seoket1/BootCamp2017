import numpy as np
import scipy.optimize as opt
from numba import jit

# 1.8
# Parameters
gamma = 2.5000
xi = 1.5000
beta = 0.9800
alpha = 0.4000
a = 0.5000
delta = 0.1000
zbar = 0.0000
rhoz = 0.900
tau = 0.0500
sigma = 0.0200

fixed_params = np.array([gamma, beta, alpha, delta, zbar, tau, xi, a])


def nume_sol(k_l_init, args = fixed_params):
    kbar, lbar = k_l_init
    gamma, beta, alpha, delta, zbar, tau, xi, a = args
    utilbar, outputbar, kbar, lbar, wbar, rbar, cbar, Tbar, investbar = nume_sol2(kbar, lbar, *args)

    error1 = beta*((rbar-delta)*(1-tau) + 1) - 1
    error2 = ( wbar * (1-tau) * (1-lbar)**xi ) / (a*(cbar**gamma)) - 1
    return abs(error1), abs(error2)

def nume_sol2(kbar, lbar, *args):
    gamma, beta, alpha, delta, zbar, tau, xi, a = args
    
    rbar = alpha * kbar ** (alpha - 1) * lbar **(1-alpha) * np.exp(zbar)**(1-alpha)
    outputbar = kbar**alpha*(lbar*np.exp(zbar))**(1-alpha)

    
    investbar = delta * kbar
    wbar = (1-alpha) * kbar**alpha*lbar**(-alpha)*np.exp(zbar)**(1-alpha)
    Tbar = tau *(wbar * lbar +(rbar - delta) * kbar)
    cbar = (1 - tau) * (wbar*lbar + (rbar - delta) * kbar) + Tbar
    utilbar = (cbar**(1-gamma) - 1) / (1 - gamma) + a*((1-lbar)**(1-xi) - 1) / (1-xi)
    
    return utilbar, outputbar, kbar, lbar, wbar, rbar, cbar, Tbar, investbar


kbar_init = 0.4
lbar_init = 0.4

k_l_bar_guess = np.array([kbar_init, lbar_init])

k_l_bar_nume = opt.fsolve(nume_sol, k_l_bar_guess)

utilbar, outputbar, kbar, lbar, wbar, rbar, cbar, Tbar, investbar = nume_sol2(k_l_bar_nume[0],k_l_bar_nume[1],  *fixed_params)
print("\n\nutilbar, outputbar, kbar, lbar, wbar, rbar, cbar, Tbar, investbar =", utilbar, outputbar, kbar, lbar, wbar, rbar, cbar, Tbar, investbar)


# Numerical dy/dx
h = 1e-7

k_d = np.zeros(8)
c_d = np.zeros(8)
r_d = np.zeros(8)
w_d = np.zeros(8)
l_d = np.zeros(8)
T_d = np.zeros(8)
y_d = np.zeros(8)
i_d = np.zeros(8)

for i in range(8):
    fixed_params[i] += h
    k_l_bar_nume_plus_h = opt.fsolve(nume_sol, k_l_bar_guess, args=(fixed_params))
    utilbar_plus_h, y_plus_h, k_plus_h, l_plus_h, w_plus_h, r_plus_h, c_plus_h, T_plus_h, i_plus_h = nume_sol2(k_l_bar_nume_plus_h[0],k_l_bar_nume_plus_h[1],  *fixed_params)
    
    k_d[i] = (k_plus_h - kbar) / h
    c_d[i] = (c_plus_h - cbar) / h
    r_d[i] = (r_plus_h - rbar) / h
    w_d[i] = (w_plus_h - wbar) / h
    l_d[i] = (l_plus_h - lbar) / h
    T_d[i] = (T_plus_h - Tbar) / h
    y_d[i] = (y_plus_h - outputbar) / h
    i_d[i] = (i_plus_h - investbar) / h
    
    # Parameters
    gamma = 2.500000
    xi = 1.500000
    beta = 0.980000
    alpha = 0.400000
    a = 0.500000
    delta = 0.100000
    zbar = 0.000000
    rhoz = 0.90000
    tau = 0.050000
    sigma = 0.020000
    fixed_params = np.array([gamma, beta, alpha, delta, zbar, tau, xi, a])
 
print("numerator order, [k_d, c_d, r_d, w_d, l_d, T_d, y_d, i_d" )
print("denominator oder, [gamma, beta, alpha, delta, zbar, tau, xi, a]") 
print("k_d :",k_d)
print("c_d :",c_d)
print("r_d :",r_d)
print("w_d :",w_d)
print("l_d :",l_d)
print("T_d :",T_d)
print("y_d :",y_d)
print("i_d :",i_d)
