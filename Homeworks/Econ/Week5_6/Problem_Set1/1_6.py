import numpy as np
import scipy.optimize as opt

# 1.6

# Parameters
gamma = 2.5
beta = 0.98
alpha = 0.40
delta = 0.10
zbar = 0
tau = 0.05
xi = 1.5
a = 0.5
fixed_params = (gamma, beta, alpha, delta, zbar, tau, xi, a)


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



