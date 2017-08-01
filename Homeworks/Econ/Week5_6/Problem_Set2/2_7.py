import numpy as np
import scipy.optimize as opt
from numba import jit
from LinApp_Deriv import LinApp_Deriv
from LinApp_FindSS import LinApp_FindSS
from LinApp_Solve import LinApp_Solve
from numba import jit
from matplotlib import pyplot as plt
###################################################################################
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

fixed_params_new = np.array([gamma, beta, alpha, delta, tau, xi, a])

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
###################################################################################
PP = np.array([[ 0.9152937 ,  0.], [-0.19192697,  0.]])
QQ = np.array([[ 0.1289971 ], [-0.01131015]])
Xbar = np.array([4.2252290288258694, 0.57979145317078085])

num_time = 1000
periods = 250
sigmaz = .02

y_simul = np.zeros((num_time, periods - 1))
c_simul = np.zeros((num_time, periods - 1))
i_simul = np.zeros((num_time, periods - 1))
l_simul = np.zeros((num_time, periods - 1))

# 10,000 Simulation.
for i in range(num_time):
    epsilon = np.random.normal(0, sigmaz, periods)
    
    X_til = np.zeros((2, periods))
    X_til[:, 0] = np.log(Xbar) - np.log(Xbar)
    
    Z_til = np.zeros(periods)
    for t in range(periods-1):
        Z_til[t+1] = rhoz * Z_til[t] + epsilon[t]
        X_til[:, t+1] = np.dot(PP, X_til[:,t]) + (QQ * Z_til[t]).T
        
    # converting back to actual values
    Xt = (Xbar*np.exp(X_til).T).T
    kt = Xt[0, :]
    lt = Xt[1, :]
    zt = Z_til + zbar # zbar = 0
    util, output, k, l, w, r, c, T, invest = nume_sol2_new(kt[1:], kt[:-1], lt[:-1], zt[:-1])

    y_simul[i, :] = output
    c_simul[i, :] = c
    i_simul[i, :] = invest
    l_simul[i, :] = l

# average of each simulation of time-series. by column
y_mean = y_simul.mean(axis = 0)
c_mean = c_simul.mean(axis = 0)
i_mean = i_simul.mean(axis = 0)
l_mean = l_simul.mean(axis = 0)

# confidence level
up = int(0.95 * num_time)
low = int(0.05 * num_time)
 
y_up = np.sort(y_simul, axis = 0)[up, :]
y_low = np.sort(y_simul, axis = 0)[low, :]

c_up = np.sort(c_simul, axis = 0)[up, :]
c_low = np.sort(c_simul, axis = 0)[low, :]

i_up = np.sort(i_simul, axis = 0)[up, :]
i_low = np.sort(i_simul, axis = 0)[low, :]

l_up = np.sort(l_simul, axis = 0)[up, :]
l_low = np.sort(l_simul, axis = 0)[low, :]
    
# plot
plt.plot(y_mean, label ="mean")
plt.plot(y_up, label ="upper limit, 95%")
plt.plot(y_low, label ="lower limit, 5%")
plt.legend()
plt.title("GDP") 
plt.show()

plt.plot(c_mean, label ="mean")
plt.plot(c_up, label ="upper limit, 95%")
plt.plot(c_low, label ="lower limit, 5%")
plt.legend()
plt.title("Consumption") 
plt.show()

plt.plot(i_mean, label ="mean")
plt.plot(i_up, label ="upper limit, 95%")
plt.plot(i_low, label ="lower limit, 5%")
plt.legend()
plt.title("Invetment") 
plt.show()

plt.plot(l_mean, label ="mean")
plt.plot(l_up, label ="upper limit, 95%")
plt.plot(l_low, label ="lower limit, 5%")
plt.legend()
plt.title("Labor Input") 
plt.show()
    
    
    
    
    
    
    
    
    
    