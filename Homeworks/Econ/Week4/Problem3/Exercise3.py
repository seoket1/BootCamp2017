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
from scipy import integrate as intgr

# Problem 3
# (a)
print("\n\n(a)\n")
usin = np.loadtxt('C:/Users/suket/Desktop/Homeworks/Econ/Week4/data/usincmoms.txt', delimiter='\t')

income = usin[:,1]/1000
frequency = usin[:,0]

def histogram(income, frequency):
    num_bins = 42
    histogram = plt.bar(left = income, height = frequency, width = 5,  edgecolor = 'black')
    histogram[40].set_height(histogram[40].get_height()/10)
    histogram[41].set_height(histogram[41].get_height()/20)
    histogram[40].set_width(50)
    histogram[41].set_width(100)
    histogram[40].set_x(200)
    histogram[41].set_x(250)
    plt.xlim(0, 350)
    plt.xlabel("Icome Class")
    plt.ylabel("Frequency as percent")
    plt.show()
    
histogram(income, frequency)
# (b)
print("\n\n(b)\n")
def data_moments(xvals):
    data = np.copy(xvals)
    data[40] = data[40]/10
    data[41] = data[41]/20
    return data

def model_moments(income, mu, sigma):
    f = lambda x: sts.lognorm.pdf(x, s = sigma, scale = np.exp(mu))
    moment = np.zeros_like(income)
    for i in range(42):
        moment[i] = intgr.quad(f, income[i] - 2.5, income[i] + 2.5)[0]
        
    return moment

def err_vec(frequency, income, mu, sigma):
    moment_data = data_moments(frequency)
    moment_model = model_moments(income, mu, sigma)
    err_vec = (moment_model - moment_data) / moment_data

    return err_vec

def criterion(params, *args):
    mu, sigma = params
    frequency, income, W = args
    err = err_vec(frequency, income, mu, sigma)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val

mu = np.log(np.sum(income * frequency))
sigma = 5
mo_frequency = np.copy(frequency)
mo_frequency[40] = mo_frequency[40] / 10
mo_frequency[41] = mo_frequency[41] / 20
W = np.diag(mo_frequency)

params_init = np.array([mu, sigma])
gmm_args = (frequency, income, W)
results = opt.minimize(criterion, params_init, args=(gmm_args), method='L-BFGS-B',\
                       bounds=((1e-10, None), (1e-10, None)))
mu_GMM, sigma_GMM = results.x

params_GMM = np.array([mu_GMM, sigma_GMM])
critic = criterion(params_GMM, frequency, income, W)

print(results)
print('mu_GMM=', mu_GMM, 'sigma_GMM=', sigma_GMM, 'critic =', critic)

# Plot the estimated curve 
model = model_moments(income, mu_GMM , sigma_GMM)
plt.plot(income, model, color = "r")

# Plot the histogram
histogram(income, frequency)

# (c)
print("\n\n(c)\n")
def model_moments_ga(income, alpha, beta):
    g = lambda x: sts.gamma.pdf(x, a = alpha, scale = beta)
    moment = np.zeros_like(income)
    for i in range(42):
        moment[i] = intgr.quad(g, income[i] - 2500, income[i] + 2500)[0]

    return moment

def err_vec_ga(frequency, income, alpha, beta):
    moment_data = data_moments(frequency)
    moment_model = model_moments_ga(income, alpha, beta)
    err_vec = (moment_model - moment_data) / moment_data

    return err_vec

def criterion_ga(params, *args):
    alpha, beta = params
    frequency, income, W = args
    err = err_vec_ga(frequency, income, alpha, beta)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val


alpha = 3
beta = 20000
mo_frequency = np.copy(frequency)
mo_frequency[40] = mo_frequency[40] / 10
mo_frequency[41] = mo_frequency[41] / 20
W = np.diag(mo_frequency)


origin_income = np.copy(income) * 1000

params_init = np.array([alpha, beta])
gmm_args = (frequency, origin_income, W)
results = opt.minimize(criterion_ga, params_init, args=(gmm_args), method='L-BFGS-B',\
                       bounds=((1e-10, None), (1e-10, None)))
alpha_GMM, beta_GMM = results.x

params_GMM = np.array([alpha_GMM, beta_GMM])
critic = criterion(params_GMM, frequency, income, W)

print(results)
print('alpha_GMM=', alpha_GMM, 'beta_GMM=', beta_GMM, 'critic =', critic)

# Plot the estimated curve 
model = model_moments_ga(income*1000, alpha_GMM , beta_GMM)
plt.plot(income, model, color = "r")

# Plot the histogram
histogram(income, frequency)

# (d)
# Plot the estimated curve 
model = model_moments(income, mu_GMM , sigma_GMM)
plt.plot(income, model, color = "red", label = "log normal")

# Plot the estimated curve 
model = model_moments_ga(income*1000, alpha_GMM , beta_GMM)
plt.plot(income, model, color = "green", label = "GA")
plt.legend()

# Plot the histogram
histogram(income, frequency)


# (e)
''' Pseudo Inverse '''
err = err_vec_ga(frequency, origin_income, alpha_GMM, beta_GMM)
omega2 = np.outer(err, err) / 42
W2 = np.linalg.pinv(omega2)
params_init = np.array([alpha_GMM, beta_GMM])
gmm_args = (frequency, origin_income, W2)

results = opt.minimize(criterion_ga, params_init, args=(gmm_args), method='L-BFGS-B',\
                       bounds=((1e-10, None), (1e-10, None)))
alpha_GMM, beta_GMM = results.x

params_GMM = np.array([alpha_GMM, beta_GMM])
critic = criterion(params_GMM, frequency, income, W)

print(results)
print('alpha_GMM=', alpha_GMM, 'beta_GMM=', beta_GMM, 'critic =', critic)

# Plot the estimated curve 
model = model_moments_ga(origin_income, alpha_GMM , beta_GMM)
plt.plot(income, model, color = "r")
plt.title("Pseudo inverse two step")
plt.ylim(0, 0.07)

# Plot the histogram
histogram(income, frequency)












