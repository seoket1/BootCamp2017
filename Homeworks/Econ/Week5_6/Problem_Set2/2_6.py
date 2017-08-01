import numpy as np
import scipy.optimize as opt
from numba import jit
from LinApp_Deriv import LinApp_Deriv
from LinApp_FindSS import LinApp_FindSS
from LinApp_Solve import LinApp_Solve

# 2.6
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

#
fixed_params_new = np.array([gamma, beta, alpha, delta, tau, xi, a])

def nume_sol_new(theta0, args = fixed_params_new):
    k2, l1, k1, l0, k0, l_1, z1, z0 = theta0
    gamma, beta, alpha, delta, tau, xi, a = args
    util1, output1, k1, l1, w1, r1, c1, T1, invest1 = nume_sol2_new(k2, k1, l1, z1, args = fixed_params_new)
    util0, output0, k0, l0, w0, r0, c0, T0, invest0 = nume_sol2_new(k1, k0, l0, z0, args = fixed_params_new)

    error2 = beta*(c1**(-gamma)*((r1-delta)*(1-tau) + 1)) - c0**(-gamma)
    error1 = ( w0 * (1-tau) * (1-l0)**xi ) / (a*(c0**gamma)) - 1

    return np.array([error1, error2])

def nume_sol2_new(k1, k0, l0, z0, args = fixed_params_new):
    gamma, beta, alpha, delta, tau, xi, a = args
    
    output0 = k0**alpha*(l0*np.exp(z0))**(1-alpha)
    r0 = alpha * k0 ** (alpha - 1) * l0 **(1-alpha) * np.exp(z0)**(1-alpha)
    
    invest0 = k1 - (1 - delta) * k0
    w0 = (1-alpha) * k0**alpha*l0**(-alpha)*np.exp(z0)**(1-alpha)
    T0 = tau *(w0 * l0 +(r0 - delta) * k0)
    c0 = (1 - tau) * (w0*l0 + (r0 - delta) * k0) + k0 + T0 - k1
    util0 = (c0**(1-gamma) - 1) / (1 - gamma) + a*((1-l0)**(1-xi) - 1) / (1-xi)
    
    return util0, output0, k0, l0, w0, r0, c0, T0, invest0

nx,ny,nz = np.array([2, 0, 1])
theta0 = np.array([kbar, lbar, kbar, lbar, kbar, lbar, zbar, zbar])
[AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, WW, TT] = LinApp_Deriv(nume_sol_new, fixed_params_new, theta0, nx, ny, nz, logX =True)

NN = rhoz
PP, QQ, UU, RR, SS, VVV = LinApp_Solve(AA,BB,CC,DD,FF,GG,HH,JJ,KK,LL,MM,WW,TT,NN,zbar,Sylv=0)

print("PP", PP)
print("QQ", QQ)