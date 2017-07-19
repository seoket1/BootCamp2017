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

# Problem 2
# (a)
print("\n\n(a)\n")

# discount factor
beta = 0.99

#get z from dataset
def get_z(w, k, alpha, rho, mu, sigma):
    z_data = np.zeros((101,1))
    z_data[0, 0] = float(mu)
    for i in range(100):
        z_data[i+1, 0] = np.log( (w[i]) / ((1 - alpha)*(k[i]**alpha)) )
    return z_data

def gen_z_pdf(w, k, alpha, rho, mu, sigma):
    z_data = get_z(w, k, alpha, rho, mu, sigma)
    z_pdf = []
    for i in range(100):
        pdf_transition = sts.norm.pdf(z_data[i+1,0], loc = rho*z_data[i] + (1-rho)*mu, scale = sigma)
        z_pdf.append(pdf_transition)
    z_pdf = np.array(z_pdf)
    return z_pdf
    
def log_lik_z(w, k, alpha, rho, mu, sigma):
    z_data = get_z(w, k, alpha, rho, mu, sigma)
    z_pdf = []
    for i in range(100):
        pdf_transition = sts.norm.logpdf(z_data[i+1,0], loc = rho*z_data[i] + (1-rho)*mu, scale = sigma)
        #pdf_transition = sts.norm.logpdf(z_data[i+1, 0], mu, scale = sigma)
        z_pdf.append(pdf_transition)
    ln_pdf_vals = np.array(z_pdf)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val

def crit_z(params, *args):
    alpha, rho, mu, sigma = params
    w, k= args
    log_lik_val = log_lik_z(w, k, alpha, rho, mu, sigma)
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val
    
# initial value
alpha =  0.42
rho = 1
mu = 9.9
sigma = 0.1
params_init = np.array([alpha, rho, mu, sigma])
mle_args = (w, k)

results = opt.minimize(crit_z, params_init, args=(mle_args), method='L-BFGS-B',\
                       bounds=((1e-10, 1 - 1e-10), (-1 + 1e-10, 1 - 1e-10), (1e-10, None), (1e-10, None)))
alpha_MLE, rho_MLE, mu_MLE, sigma_MLE = results.x

print('\nalpha_MLE=', alpha_MLE, ' rho_MLE=', rho_MLE, 'mu_MLE=', mu_MLE, 'sigma_MLE=', sigma_MLE)

log_likelihood = log_lik_z(w, k, alpha_MLE, rho_MLE, mu_MLE, sigma_MLE)
print('\nlog_likelihood=', log_likelihood)

results
vcv_mle = results.hess_inv(np.eye(4))
print('VCV(MLE) = ', vcv_mle)

'''
vcv_mle = results.hess_inv
print('VCV(MLE) = ', vcv_mle)
print(results)
# alpha_MLE, rho_MLE, mu_MLE, sigma_MLE = 0.42, 0.1, 9.9, 1
'''
'''
# test for MLE
epsilon = np.zeros((100,1))
for i in range(100):
    epsilon[i] = np.random.normal(loc = 0, scale = sigma_MLE)

test = np.zeros((100,1))
z = np.zeros((101,1))
z[0,0] = mu_MLE
for i in range(100):
    z[i+1,0] = rho_MLE * z[i] + (1-rho_MLE)*mu_MLE  + epsilon[i]
    test[i]  = w[i] - (1-alpha_MLE)*np.exp(z[i+1,0])*k[i]**alpha

print("\n", test)
'''


# (b)
print("\n\n(b)\n")

#get z from dataset
def get_z2(r, k, alpha, rho, mu, sigma):
    z_data = np.zeros((101,1))
    z_data[0, 0] = float(mu)
    for i in range(100):
        z_data[i+1, 0] = (1-alpha)*np.log(k[i]) + np.log(1/alpha) + np.log(r[i])
    return z_data

def gen_z_pdf2(r, k, alpha, rho, mu, sigma):
    z_data = get_z2(r, k, alpha, rho, mu, sigma)
    z_pdf = []
    for i in range(100):
        pdf_transition = sts.norm.pdf(z_data[i+1,0], loc = rho*z_data[i] + (1-rho)*mu, scale = sigma)
        z_pdf.append(pdf_transition)
    z_pdf = np.array(z_pdf)
    return z_pdf
    
def log_lik_z2(r, k, alpha, rho, mu, sigma):
    z_data = get_z2(r, k, alpha, rho, mu, sigma)
    z_pdf = []
    for i in range(100):
        pdf_transition = sts.norm.logpdf(z_data[i+1,0], loc = rho*z_data[i] + (1-rho)*mu, scale = sigma)
        z_pdf.append(pdf_transition)
    ln_pdf_vals = np.array(z_pdf)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val

def crit_z2(params, *args):
    alpha, rho, mu, sigma = params
    r, k= args
    log_lik_val = log_lik_z2(r, k, alpha, rho, mu, sigma)
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val
    
# initial value
alpha =  0.42
rho = 1
mu = 17
sigma = 5
 
params_init = np.array([alpha, rho, mu, sigma])
mle_args = (r, k)

results = opt.minimize(crit_z2, params_init, args=(mle_args), method='L-BFGS-B',\
                       bounds=((1e-10, 1 - 1e-10), (-1 + 1e-10, 1 - 1e-10), (1e-10, None), (1e-10, None)))
alpha_MLE_b, rho_MLE_b, mu_MLE_b, sigma_MLE_b = results.x

print(results)

print('\nalpha_MLE=', alpha_MLE_b, ' rho_MLE=', rho_MLE_b, 'mu_MLE=', mu_MLE_b, 'sigma_MLE=', sigma_MLE_b)

log_likelihood = log_lik_z2(r, k, alpha_MLE_b, rho_MLE_b, mu_MLE_b, sigma_MLE_b)
print('\nlog_likelihood=', log_likelihood)

vcv_mle2 = results.hess_inv(np.eye(4))
print('VCV(MLE) = ', vcv_mle2)


# (c)
print("\n\n(c)\n")

alpha_MLE, rho_MLE, mu_MLE, sigma_MLE
z_data_c = get_z(w, k, alpha_MLE, rho_MLE, mu_MLE, sigma_MLE)
z_star =  (1-alpha_MLE)*np.log(7500000) + np.log(1/alpha_MLE) 

num = 100000
qr = np.random.uniform(z_star, max(z_data_c), num)
v_qr = sts.norm.pdf(qr, loc = rho_MLE*10 + (1-rho_MLE)*mu_MLE, scale = sigma_MLE)

mcint = (max(z_data_c) - z_star) * (np.sum(v_qr)/num)

approx_dif = (mcint - (len(z_data_c[z_data_c > z_star]) / np.shape(z_data_c)[0]))
print("Probability more than \"r > 1\"----->>", mcint)






'''

(a)


alpha_MLE= 0.457509740793  rho_MLE= 0.720493225849 mu_MLE= 9.52281143774 sigma_MLE= 0.0919960686003

log_likelihood= 96.7069080647
VCV(MLE) =  [[  1.51156913e+01  -2.36913700e+01  -2.15416226e+02  -8.12357241e-01]
 [ -2.36913700e+01   4.02509812e+01   3.37273701e+02   8.95233561e-01]
 [ -2.15416226e+02   3.37273701e+02   3.06997571e+03   1.16211943e+01]
 [ -8.12357241e-01   8.95233561e-01   1.16211943e+01   9.02239012e-02]]


(b)

      fun: -96.706908041367427
 hess_inv: <4x4 LbfgsInvHessProduct with dtype=float64>
      jac: array([ 0.02636824, -0.00052722,  0.00147509, -0.00597709])
  message: b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
     nfev: 350
      nit: 47
   status: 0
  success: True
        x: array([ 0.45749206,  0.72050429,  9.37072991,  0.09199598])

alpha_MLE= 0.457492063757  rho_MLE= 0.720504293511 mu_MLE= 9.37072990581 sigma_MLE= 0.0919959782314

log_likelihood= 96.7069080414
VCV(MLE) =  [[  1.28817433e+00  -5.32358432e+00  -2.33234473e+01  -3.17381590e-01]
 [ -5.32358432e+00   2.59337351e+01   9.64073491e+01   1.67159559e+00]
 [ -2.33234473e+01   9.64073491e+01   4.22291108e+02   5.74832082e+00]
 [ -3.17381590e-01   1.67159559e+00   5.74832082e+00   1.11440783e-01]]


(c)

Probability more than "r > 1"----->> [ 0.68219785]
'''

















