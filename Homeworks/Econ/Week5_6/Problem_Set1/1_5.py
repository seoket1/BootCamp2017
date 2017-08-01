import numpy as np
import scipy.optimize as opt

# 1.5

# Parameters
gamma = 2.5
beta = 0.98
alpha = 0.40
delta = 0.10
zbar = 0
tau = 0.05

lbar = 1

def algeb_sol(gamma, beta, alpha, delta, zbar, tau, lbar):
    rbar = (1/beta -1)*(1/(1-tau)) + delta
    kbar = (rbar * 1/(alpha*np.exp(zbar*(1-alpha))))**(1/(alpha - 1))
    outputbar = kbar**alpha*(lbar*np.exp(zbar))**(1-alpha)
    investbar = delta * kbar
    wbar = (1-alpha)*kbar**alpha*np.exp(zbar)**(1-alpha)
    Tbar = tau *(wbar +(rbar - delta) * kbar)
    cbar = (1 - tau) * (wbar*lbar + (rbar - delta) * kbar) + Tbar
    utilbar = (cbar**(1-gamma) - 1) / (1 - gamma)
    
    return utilbar, outputbar, kbar, lbar, wbar, rbar, cbar, Tbar, investbar

utilbar, outputbar, kbar, lbar, wbar, rbar, cbar, Tbar, investbar = algeb_sol(gamma, beta, alpha, delta, zbar, tau, lbar)
print("utilbar, outputbar, kbar, lbar, wbar, rbar, cbar, Tbar, investbar =", algeb_sol(gamma, beta, alpha, delta, zbar, tau, lbar))

def nume_sol(k_init, *args):
    kbar = k_init
    gamma, beta, alpha, delta, zbar, tau, lbar = args
    
    rbar = alpha * kbar ** (alpha - 1) * np.exp(zbar)**(1-alpha)
    error = beta*((rbar-delta)*(1-tau) + 1) - 1
    
    return abs(error)

def nume_sol2(kbar, *args):
    gamma, beta, alpha, delta, zbar, tau, lbar = args
    
    rbar = alpha * kbar ** (alpha - 1) * np.exp(zbar)**(1-alpha)
    outputbar = kbar**alpha*(lbar*np.exp(zbar))**(1-alpha)
    investbar = delta * kbar
    wbar = (1-alpha)*kbar**alpha*np.exp(zbar)**(1-alpha)
    Tbar = tau *(wbar +(rbar - delta) * kbar)
    cbar = (1 - tau) * (wbar*lbar + (rbar - delta) * kbar) + Tbar
    utilbar = (cbar**(1-gamma) - 1) / (1 - gamma)
    
    return utilbar, outputbar, kbar, lbar, wbar, rbar, cbar, Tbar, investbar


kbar_init = 1
fixed_params = (gamma, beta, alpha, delta, zbar, tau, lbar)
kbar_guess = np.array([kbar_init])

kbar_nume = opt.minimize(nume_sol, kbar_guess, args=(fixed_params))


utilbar, outputbar, kbar, lbar, wbar, rbar, cbar, Tbar, investbar = nume_sol2(kbar_nume.x, *fixed_params)
print("\n\nutilbar, outputbar, kbar, lbar, wbar, rbar, cbar, Tbar, investbar =", utilbar, outputbar, kbar, lbar, wbar, rbar, cbar, Tbar, investbar)



