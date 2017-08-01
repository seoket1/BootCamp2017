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

periods = 41
sigmaz = .02
epsilon = np.zeros(periods)
epsilon[0] = 0.001
Z_til = np.zeros(periods)
Z_til[0] = 0
X_til = np.zeros((2, periods))
X_til[:, 0] = X_til[:, 0] = np.log(Xbar) - np.log(Xbar)

for t in range(periods-1):
    Z_til[t+1] = rhoz * Z_til[t] + epsilon[t]
    X_til[:, t+1] = np.dot(PP, X_til[:,t]) + (QQ * Z_til[t]).T
    
# converting back to actual values
Xt = (Xbar*np.exp(X_til).T).T
kt = Xt[0, :]
lt = Xt[1, :]
zt = Z_til + zbar # zbar = 0
util, output, k, l, w, r, c, T, invest = nume_sol2_new(kt[1:], kt[:-1], lt[:-1], zt[:-1])

y_simul = output
c_simul = c
i_simul = invest
l_simul = l

# plot
plt.plot(y_simul)
plt.legend()
plt.title("GDP") 
plt.show()

plt.plot(c_simul)
plt.legend()
plt.title("Consumption") 
plt.show()

plt.plot(i_simul)
plt.legend()
plt.title("Invetment") 
plt.show()

plt.plot(l_simul)
plt.legend()
plt.title("Labor Input") 
plt.show()
    
    
    
    
    
    
    
    
    
    