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

clms = np.loadtxt('C:/Users/suket/Desktop/Homeworks/Econ/Week4/data/clms.txt')

# This next command is specifically for Jupyter Notebook
#count, bins, ignored = plt.hist(clms, 30, normed=True, edgecolor='black',
#                                linewidth=1.2)

# (a)
print("\n\n(a)\n")
mean = np.mean(clms)
median = np.median(clms)
maximum = np.max(clms)
minimum = np.min(clms)
std = np.std(clms)

print("Mean =", mean)
print("\nMedian", median)
print("\nMax =", maximum)
print("\nMin =", minimum)
print("\nStd =", std)

def First_Histogram():
    num_bins = 1000
    weights = (1 / clms.shape[0]) * np.ones_like(clms)
    n, bin_cuts, patches = plt.hist(clms, num_bins, weights=weights, edgecolor='black', linewidth=1.2)
    plt.title('First Histogram', fontsize=17)
    plt.xlabel(r'value of monthly expenditure')
    plt.ylabel(r'percentage of observations')
    plt.show()

def Second_Histogram():
    num_bins = 100
    clms_800 = clms[clms <= 800]
    weights = (1 / (8*clms.shape[0])) * np.ones_like(clms_800)
    n, bin_cuts, patches = plt.hist(clms_800, num_bins, weights=weights, edgecolor='black', linewidth=1.2)
    plt.title('Second Histogram', fontsize=17)
    plt.xlabel(r'value of monthly expenditure')
    plt.ylabel(r'percentage of observations')
    plt.ylim(0, 0.005)
    plt.show()

First_Histogram()
Second_Histogram()

# (b)
print("\n\n(b)\n")
def gamma_pdf(clms, alpha0, beta0):
    return sts.gamma.pdf(clms, a = alpha0, scale = beta0)

def log_lik_gamma(clms, alpha0, beta0):
    pdf_vals = gamma_pdf(clms, alpha0, beta0)
    pdf_vals = pdf_vals[pdf_vals>1e-5000]
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val

def crit_gamma(params, *args):
    alpha0, beta0 = params
    xvals = args
    log_lik_val = log_lik_gamma(xvals, alpha0, beta0)
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val

beta0_init =  np.var(clms)/np.mean(clms)
alpha0_init = np.mean(clms)/beta0_init
params_init = np.array([alpha0_init, beta0_init])
mle_args = (clms)
results = opt.minimize(crit_gamma, params_init, args=(mle_args), method='L-BFGS-B',\
                       bounds=((1e-10, None), (1e-10, None)))
alpha0_MLE, beta0_MLE = results.x
alpha0_MLE_ga, beta0_MLE_ga = alpha0_MLE, beta0_MLE
print('alpha0_MLE=', alpha0_MLE, ' beta0_MLE=', beta0_MLE)

log_likelihood = log_lik_gamma(clms, alpha0_MLE, beta0_MLE)
print('log_likelihood=', log_likelihood)

# Plot the MLE estimated distribution
dist_pts = np.linspace(0, 800, 5000)
plt.plot(dist_pts, gamma_pdf(dist_pts, alpha0_MLE, beta0_MLE), color='r', label = 'gamma approx')
plt.legend(loc='upper right')

# Plot Histogram
Second_Histogram()

# (c)
print("\n\n(c)\n")
def gen_gamma_pdf(clms, alpha0, beta0, m):
    return sts.gengamma.pdf(clms, a = alpha0, scale = beta0, c = m)
                            
def log_lik_gen_gamma(clms, alpha0, beta0, m):
    ln_pdf_vals = sts.gengamma.logpdf(clms, a = alpha0, scale = beta0, c = m)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val

def crit_gen_gamma(params, *args):
    alpha0, beta0, m = params
    xvals= args
    log_lik_val = log_lik_gen_gamma(xvals, alpha0, beta0, m)
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val

m_init = 4
beta0_init =  beta0_MLE
alpha0_init = alpha0_MLE
params_init = np.array([alpha0_init, beta0_init, m_init])
mle_args = (clms)
results = opt.minimize(crit_gen_gamma, params_init, args=(mle_args), method='L-BFGS-B',\
                       bounds=((1e-10, None), (1e-10, None), (1e-10, None)))

alpha0_MLE, beta0_MLE, m_MLE = results.x

print('alpha0_MLE=', alpha0_MLE, ' beta0_MLE=', beta0_MLE, 'm_MLE=', m_MLE)

log_likelihood = log_lik_gamma(clms, alpha0_MLE, beta0_MLE)
print('log_likelihood=', log_likelihood)

'''
m_init = 1
alpha0_MLE= 0.222274516597  beta0_MLE= 21911.0646302 m_MLE= 0.997656451756
log_likelihood= -82076.4824274

m_init = 4
alpha0_MLE= 0.22227209114  beta0_MLE= 21911.0648185 m_MLE= 0.997674346426
log_likelihood= -82076.4821575
'''

# Plot the MLE estimated distribution
dist_pts = np.linspace(0, 800, 5000)
plt.plot(dist_pts, gen_gamma_pdf(dist_pts, alpha0_MLE, beta0_MLE, m_MLE), color='r', label = 'general gamma approx')
plt.legend(loc='upper right')

# Plot Histogram
Second_Histogram()

# (d)
print("\n\n(d)\n")
def gen_beta2_pdf(y, a, b, p, q):
    B = scipy.special.beta(p, q)
    denom1 =(1 + (y/b)**a )**(p+q)   
    if np.any( denom1 == np.inf ) == True:
        denom1[denom1 == np.inf] = 1e+10
        denom2 = ( b**(a*p) * B * denom1)
        GB2_pdf = a*(y)**(a*p-1) / denom2
    else:
        GB2_pdf = a*(y)**(a*p-1) / ( b**(a*p) * B * (1 + (y/b)**a )**(p+q))
    return GB2_pdf # sts.betaprime.pdf(y, p, q, loc=0, scale = b) #GB2_pdf

    #gen_beta2_pdf(clms, 1, 2, 3, 4) 
    #sts.betaprime.pdf(clms, 3, 4, loc=0, scale = 2)
def log_lik_gen_beta2(clms, a, b, p, q):
    pdf_vals = gen_beta2_pdf(clms, a, b, p, q)
    pdf_vals = pdf_vals[pdf_vals>1e-20]
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val

def crit_gen_beta2(params, *args):
    a, b, p, q = params
    xvals= args
    log_lik_val = log_lik_gen_beta2(xvals, a, b, p, q)
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val

p_init = alpha0_MLE/m_MLE
q_init = 1
b_init = q_init**(1/m_MLE) * beta0_MLE
a_init = m_MLE
params_init = np.array([a_init, b_init, p_init, q_init])
mle_args = (clms)
results = opt.minimize(crit_gen_beta2, params_init, args=(mle_args), method='L-BFGS-B',\
                       bounds=((1e-10, None), (1e-10, None), (1e-10, None), (1e-10, None)))


a_MLE, b_MLE, p_MLE, q_MLE = results.x

print('a_MLE=', a_MLE, ' b_MLE=', b_MLE, 'p_MLE=', p_MLE, 'q_MLE=', q_MLE)

log_likelihood = log_lik_gen_beta2(clms, a_MLE, b_MLE, p_MLE, q_MLE)
print('log_likelihood=', log_likelihood)


'''
q_init = 1
a_MLE= 0.11404108258  b_MLE= 21913.9465417 p_MLE= 54.1132240339 q_MLE= 92.8968411452
log_likelihood= -74862.3422261

q_init = 1e+12
a_MLE= 0.952080847491  b_MLE= 2.33670480585e+16 p_MLE= 0.212597370891 q_MLE= 1e+12
log_likelihood= -80239.932754
'''

# Plot the MLE estimated distribution
dist_pts = np.linspace(1e-100, 800, 5000)
plt.plot(dist_pts, gen_beta2_pdf(dist_pts, a_MLE, b_MLE, p_MLE, q_MLE), color='r', label = 'GB2 approx')
plt.legend(loc='upper right')

# Plot Histogram
Second_Histogram()

# (e)
print("\n\n(e)\n")
log_lik_h0 = log_lik_gen_beta2(clms, a_MLE, b_MLE, p_MLE, q_MLE)
log_lik_mle = log_lik_gamma(clms, alpha0_MLE, beta0_MLE)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 4)
print('chi squared of H0 with 4 degrees of freedom p-value (gamma) = ', pval_h0)


log_lik_h0 = log_lik_gen_beta2(clms, a_MLE, b_MLE, p_MLE, q_MLE)
log_lik_mle = log_lik_gen_gamma(clms, alpha0_MLE, beta0_MLE, m_MLE)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 4)
print('chi squared of H0 with 4 degrees of freedom p-value (general gamma) = ', pval_h0)

#(f)
print("\n\n(f)\n")
##################################################
num = 100000
qr = np.random.uniform(1000, max(clms), num)
v_qr = gen_beta2_pdf(qr, a_MLE, b_MLE, p_MLE, q_MLE)

mcint = (max(clms) - 1000) * (np.sum(v_qr)/num)
approx_dif = (mcint - (len(clms[clms > 1000])/np.shape(clms)[0]))
print("Probability more than $1000(GB2) =", mcint)
print("Difference between Original Dataset =", approx_dif)

##################################################
num = 100000
qr = np.random.uniform(1000, max(clms), num)
v_qr = gamma_pdf(qr, alpha0_MLE_ga, beta0_MLE_ga)

mcint = (max(clms) - 1000) * (np.sum(v_qr)/num)
approx_dif = (mcint - (len(clms[clms > 1000])/np.shape(clms)[0]))
print("Probability more than $1000(GA) =", mcint)
print("Difference between Original Dataset =", approx_dif)
















