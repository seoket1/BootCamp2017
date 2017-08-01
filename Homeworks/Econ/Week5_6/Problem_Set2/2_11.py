import numpy as np
import scipy.optimize as opt
from numba import jit
from LinApp_Deriv import LinApp_Deriv
from LinApp_FindSS import LinApp_FindSS
from LinApp_Solve import LinApp_Solve
from numba import jit
from matplotlib import pyplot as plt
###################################################################################
'''Household parameters'''
S = int(3)
beta_annual = 0.96
beta = beta_annual **20
sigma = 3
nvec = np.array([1.0,1.0,0.2])
L=np.sum(nvec) # Market Clearing

'''Firm parameters'''
A = 1
alpha = 0.35
delta_annual = 0.05
delta = 1- ((1-delta_annual)**20)

def get_rt(L, K, z, alpha, A, delta):
    rt = alpha * np.exp(z) * A * (L/K)**(1-alpha) - delta
    return rt
 
def get_wt(L, K, z, alpha, A, delta):
    wt = (1-alpha) * np.exp(z) * A * (K/L)**(alpha)
    return wt

def get_ct(bt, btp1, rt, wt, ls):
    ct = (1 + rt) * bt + (ls * wt) - btp1
    return ct

###################################################################################
PP = np.array([[ 0.03530577,  0.63635012], [ 0.38029487,  0.16164297]])
QQ = np.array([[ 1.84670834], [ 1.36346484]])
b_2 =  0.0193127352389
b_3 =  0.0584115908788
Xinit = ([0.8*b_2, 1.1*b_3])
Xbar =([b_2, b_3])

num_time = 10000
periods = 250
rhoz = 0.9**20
sigmaz = 0.02
zbar = 0

y_simul = np.zeros((num_time, periods - 1))
c_simul = np.zeros((num_time, periods - 1))
i_simul = np.zeros((num_time, periods - 1))

# 10,000 Simulation.
for i in range(num_time):
    epsilon = np.random.normal(0, rhoz, periods)
    
    X_til = np.zeros((2, periods))
    X_til[:, 0] = np.log(Xinit) - np.log(Xbar)
    
    Z_til = np.zeros(periods)
    for t in range(periods-1):
        Z_til[t+1] = rhoz * Z_til[t] + epsilon[t]
        X_til[:, t+1] = np.dot(PP, X_til[:,t]) + (QQ * Z_til[t]).T
        
    # converting back to actual values
    Xt = (Xbar*np.exp(X_til).T).T
    k2 = Xt[0, :]
    k3 = Xt[1, :]
    zt = Z_til + zbar # zbar = 0
    
    K = k2 + k3
    L = 2.2
    rt = get_rt(L, K, zt, alpha, A, delta)
    wt = get_wt(L, K, zt, alpha, A, delta)
    c1 = get_ct(0.0, k2[1:], 0.0, wt[:-1], 1)
    c2 = get_ct(k2[:-1], k3[1:], rt[:-1], wt[:-1], 1)
    c3 = get_ct(k3[:-1], 0.0, rt[-1], wt[-1], 0.2)
    c = c1 + c2 + c3
    
    invest = K[1:] - (1 - delta) * K[:-1]
    
    output = np.exp(zt[1:])*A*K[1:]**alpha*L**(1-alpha)
    
    y_simul[i, :] = output
    c_simul[i, :] = c
    i_simul[i, :] = invest


# average of each simulation of time-series. by column
y_mean = y_simul.mean(axis = 0)
c_mean = c_simul.mean(axis = 0)
i_mean = i_simul.mean(axis = 0)

# confidence level
up = int(0.95 * num_time)
low = int(0.05 * num_time)
 
y_up = np.sort(y_simul, axis = 0)[up, :]
y_low = np.sort(y_simul, axis = 0)[low, :]

c_up = np.sort(c_simul, axis = 0)[up, :]
c_low = np.sort(c_simul, axis = 0)[low, :]

i_up = np.sort(i_simul, axis = 0)[up, :]
i_low = np.sort(i_simul, axis = 0)[low, :]


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

    
    
    
    
    
    
    
    