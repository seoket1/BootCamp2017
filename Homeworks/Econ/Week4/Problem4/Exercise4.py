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

def get_z2(r, k, alpha, beta, rho, mu):
    z_data = (1-alpha)*np.log(k) + np.log(1/alpha) + np.log(r)
    return z_data

def data_moments(c, k, w, r, alpha, beta, rho, mu):
    z = get_z2(r, k, alpha, beta, rho, mu)

    moment1 = np.mean(z[1:]- rho*z[:-1] - (1 - rho)*mu)
    moment2 = np.mean((z[1:] - rho*z[:-1] - (1 - rho)*mu) * z[1:])
    moment3 = np.mean(beta * alpha * np.exp(z[1:]) * k[1:]**(alpha - 1) * c[:-1] / c[1:] - 1)
    moment4 = np.mean((beta * alpha * np.exp(z[1:]) * k[1:]**(alpha - 1) * c[:-1] / c[1:] - 1) * w[1:])
    
    return moment1, moment2, moment3, moment4

def err_vec(c, k, w, r, alpha, beta, rho, mu):
    moment_data = data_moments(c, k, w, r, alpha, beta, rho, mu)
    moment_model = np.zeros(4)
    err_vec = moment_model - moment_data

    return err_vec

def criterion(params, *args):
    alpha, beta, rho, mu = params
    c, k, w, r, W = args
    err = err_vec(c, k, w, r, alpha, beta, rho, mu)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val

alpha =  0.42
beta = 0.97
rho = 0
mu = 1

W = np.eye(4)

params_init = np.array([alpha, beta, rho, mu])
gmm_args = (c, k, w, r, W)

results = opt.minimize(criterion, params_init, args=(gmm_args), method='TNC', bounds=((1e-10, 1 - 1e-10), (1e-10, 1 - 1e-10), (-1 + 1e-10, 1 - 1e-10), (1e-10, None)))
alpha_GMM, beta_GMM, rho_GMM, mu_GMM = results.x

print(results)

print('\nalpha_GMM=', alpha_GMM, ' beta_GMM=', beta_GMM, 'rho_GMM=', rho_GMM, 'mu_GMM=', mu_GMM)


'''
params_GMM = np.array([alpha_GMM, rho_GMM, mu_GMM, sigma_GMM])
critic = criterion(params_GMM, gmm_args)

'''

'''
alpha =  0.42
beta = 0.96
rho = 0
mu = 9.9


     fun: 2.2304078702973298
     jac: array([ -5.56065274e+02,  -1.06262110e-01,  -4.39116801e+00,
        -3.01358631e+01])
 message: 'Converged (|x_n-x_(n-1)| ~= 0)'
    nfev: 10
     nit: 3
  status: 2
 success: True
       x: array([  4.19999996e-01,   9.89999995e-01,   1.59258810e-13,
         9.90000000e+00])

alpha_GMM= 0.419999995898  beta_GMM= 0.989999995 rho_GMM= 1.59258810433e-13 mu_GMM= 9.90000000003
'''

'''
alpha =  0.42
beta = 0.97
rho = 0
mu = 9

     fun: 0.0028502917914412038
     jac: array([ -1.75334601e+01,   1.06392173e+06,  -3.66196474e-03,
        -9.75882696e-01])
 message: 'Converged (|x_n-x_(n-1)| ~= 0)'
    nfev: 65
     nit: 12
  status: 2
 success: True
       x: array([ 0.46565576,  0.99      ,  0.01312137,  9.217306  ])

alpha_GMM= 0.465655761447  beta_GMM= 0.98999999978 rho_GMM= 0.0131213684588 mu_GMM= 9.21730600318
'''



