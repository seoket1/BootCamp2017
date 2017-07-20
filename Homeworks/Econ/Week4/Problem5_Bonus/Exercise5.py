from scipy import linalg as la
from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt
import cmath
import math
from scipy.sparse import dok_matrix
import scipy.stats as sts
import sympy as sy
from matplotlib import pyplot as plt
from numba import jit
from autograd import grad
import autograd.numpy as anp
from autograd import jacobian
import scipy.optimize as opt
import scipy.special

bm = np.loadtxt('C:/Users/suket/Desktop/Homeworks/Econ/Week4/data/MacroSeries.txt', delimiter=',')
# data assignin first
c = bm[:,0]
k = bm[:,1]
w = bm[:,2]
r = bm[:,3]

# Problem 4
# (a)
print("\n\n(a)\n")
def data_moments(c, k, w, r, alpha, beta, rho, mu, sigma):
    moment_c = np.mean(c)
    moment_k = np.mean(k)
    moment_c_var = np.var(c)
    moment_k_var = np.var(k)
    moment_c_k_corr = np.correlate(c, k)
    moment_k0_k1_corr = np.correlate(k[0:-1], k[1:])
    
    return moment_c, moment_k, moment_c_var, moment_k_var, moment_c_k_corr, moment_k0_k1_corr 

def smm_moments(k, alpha, beta, rho, mu, sigma, T, S):
    zz = np.zeros((S, T))
    zz[:, 0] = mu
    for s in range(S):
        for t in range(T - 1):
            zz[s, t+1] = zz[s, t] * rho + (1 - rho) * mu + np.random.normal(loc = 0, scale = sigma)
    
    kk = np.zeros((S, T))
    kk[:, 0] = np.mean(k)
    for s in range(S):
        for t in range(T-1):
            kk[s, t+1] = alpha * beta * np.exp(zz[s, t]) * kk[s, t]**alpha
    ww = (1-alpha)*np.exp(zz)*kk**alpha
    rr = alpha * np.exp(zz) * kk**(alpha - 1)        
    cc = -kk[:, 1:] + ww[:, :-1] + rr[:, :-1]*kk[:, :-1]
    
    moment_cc = cc.mean(axis = 1)
    moment_kk = kk.mean(axis = 1)
    moment_c_var = cc.var(axis = 1)
    moment_k_var = kk.var(axis = 1)
    moment_c_k_corr = np.zeros(S)
    for s in range(S):
        temp0 = cc[s,:]
        temp1 = kk[s,:-1]
        moment_c_k_corr[s] = np.correlate(temp0, temp1)
        
    moment_k0_k1_corr = np.zeros(S)
    for s in range(S):
        temp_0 = kk[s, :-1]
        temp_1 = kk[s, 1:]
        moment_k0_k1_corr[s] = np.correlate(temp_0, temp_1)
    
    model_moment_cc = np.mean(moment_cc)
    model_moment_kk = np.mean(moment_kk)
    model_moment_c_var = np.mean(moment_c_var)
    model_moment_k_var = np.mean(moment_k_var)
    model_moment_c_k_corr = np.mean(moment_c_k_corr)
    model_moment_k0_k1_corr = np.mean(moment_k0_k1_corr)
    return model_moment_cc, model_moment_kk, model_moment_c_var, model_moment_k_var, model_moment_c_k_corr, model_moment_k0_k1_corr

def err_vec(c, k, w, r, alpha, beta, rho, mu, sigma):
    moment_data = data_moments(c, k, w, r, alpha, beta, rho, mu, sigma)
    moment_model = smm_moments(k, alpha, beta, rho, mu, sigma, T, S)
    moment_data = np.array(moment_data)
    moment_model = np.array(moment_model)
    err_vec = (moment_model - moment_data)/moment_data
    return err_vec

def criterion(params, *args):
    alpha, beta, rho, mu, sigma = params
    c, k, w, r, W, T, S = args
    err = err_vec(c, k, w, r, alpha, beta, rho, mu, sigma)
    crit_val = np.dot(np.dot(err.T, W), err) 
    return crit_val

alpha =  0.3
beta = 0.98
rho = 0.6
mu = 0.2
sigma = 0.8

W = np.eye(6)
T = 100
S = 1000

params_init = np.array([alpha, beta, rho, mu, sigma])
smm_args = (c, k, w, r, W, T, S)

bds=((0.01, 0.99), (0.01, 0.99), (-0.99, 0.99), (-0.5, 1), (0.001, 1))
results = opt.minimize(criterion, params_init, args=(smm_args), method='TNC', bounds = bds)
alpha_SMM, beta_SMM, rho_SMM, mu_SMM, sigma_SMM = results.x

print(results)
print('\nalpha_SMM=', alpha_SMM, ' beta_SMM=', beta_SMM, 'rho_SMM=', rho_SMM, 'mu_SMM=', mu_SMM, 'sigma_SMM=', sigma_SMM)
print(err_vec(c, k, w, r, alpha_SMM, beta_SMM, rho_SMM, mu_SMM, sigma_SMM))

'''
      fun: 5.6260765239141959
 hess_inv: <5x5 LbfgsInvHessProduct with dtype=float64>
      jac: array([ 0.02393472,  0.23641578,  0.01028919,  0.16349082,  0.01474971])
  message: b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
     nfev: 78
      nit: 1
   status: 0
  success: True
        x: array([  4.20000302e-01,   9.70000087e-01,  -3.22296185e-07,
         3.99999427e-01,   5.00000410e-01])

alpha_SMM= 0.420000301664  beta_SMM= 0.970000086749 rho_SMM= -3.2229618524e-07 mu_SMM= 0.399999426847 sigma_SMM= 0.500000409651
[-0.99999926 -0.9899993  -1.         -0.80372996 -0.99999938 -0.99999939]

     fun: 5.626080522410227
     jac: array([ 0.16111912, -0.20344801,  0.05755236,  0.02669269, -0.28161136])
 message: 'Converged (|f_n-f_(n-1)| ~= 0)'
    nfev: 26
     nit: 1
  status: 1
 success: True
       x: array([ 0.29999998,  0.97999997,  0.59999993,  0.19999992,  0.79999997])

alpha_SMM= 0.29999998402  beta_SMM= 0.979999971542 rho_SMM= 0.599999927015 mu_SMM= 0.199999916553 sigma_SMM= 0.7999999684
[-0.99999979 -0.98999988 -1.         -0.80372973 -0.99999991 -0.99999995]
'''

