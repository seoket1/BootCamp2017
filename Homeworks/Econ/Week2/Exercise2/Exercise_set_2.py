import numpy as np
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import brentq
import quantecon as qe
import time
from numba import jit


# Exercise 2


def bellman_operator(w, grid, beta, u, f, shocks, Tw=None, Tw_95=None, Tw_H=None, compute_policy=0):
    """
    The approximate Bellman operator, which computes and returns the
    updated value function Tw on the grid points.  An array to store
    the new set of values Tw is optionally supplied (to avoid having to
    allocate new arrays at each iteration).  If supplied, any existing data in 
    Tw will be overwritten.

    Parameters
    ----------
    w : array_like(float, ndim=1)
        The value of the input function on different grid points
    grid : array_like(float, ndim=1)
        The set of grid points
    beta : scalar
        The discount factor
    u : function
        The utility function
    f : function
        The production function
    shocks : numpy array
        An array of draws from the shock, for Monte Carlo integration (to
        compute expectations).
    Tw : array_like(float, ndim=1) optional (default=None)
        Array to write output values to
    compute_policy : Boolean, optional (default=False)
        Whether or not to compute policy function

    """
    # === Apply linear interpolation to w === #
    w_func = lambda x: np.interp(x, grid, w)

    # == Initialize Tw if necessary == #
    if Tw is None:
        Tw = np.empty_like(w)
    
    if Tw_95 is None:
        Tw_95 = np.empty_like(w)
        
    if Tw_H is None:
        Tw_H = np.empty_like(w)

    if compute_policy:
        sigma = np.empty_like(w)
        sigma_95 = np.empty_like(w)
        sigma_H = np.empty_like(w)

    # == set Tw[i] = max_c { u(c) + beta E w(f(y  - c) z)} == #
    for i, y in enumerate(grid):
        def objective(c):
            return - u(c) - beta * np.mean(w_func(f(y - c) * shocks)) # Bellman
        
        c_star = fminbound(objective, 1e-10, y)  
        c_star_95 = 0.95 * y
        c_star_H = scipy.stats.lognorm.cdf(y, 0.5) * y
        
        if compute_policy:
            sigma[i] = c_star
            sigma_95[i] = c_star_95
            sigma_H[i] = c_star_H
            
        Tw[i] = - objective(c_star)
        Tw_95[i] = - objective(c_star_95)
        Tw_H[i] = - objective(c_star_H)

    if compute_policy:
        return Tw, Tw_95, Tw_H, sigma, sigma_95, sigma_H
    else:
        return Tw, Tw_95, Tw_H
    
#################################################################################

class LogLinearOG:
    """
    Log linear optimal growth model, with log utility, CD production and
    multiplicative lognormal shock, so that

        y = f(k, z) = z k^alpha

    with z ~ LN(mu, s).

    The class holds parameters and true value and policy functions.
    """

    def __init__(self, alpha=0.4, beta=0.96, mu=0, s=0.1):

        self.alpha, self.beta, self.mu, self.s = alpha, beta, mu, s 

        # == Some useful constants == #
        self.ab = alpha * beta
        self.c1 = np.log(1 - self.ab) / (1 - beta)
        self.c2 = (mu + alpha * np.log(self.ab)) / (1 - alpha)
        self.c3 = 1 / (1 - beta)
        self.c4 = 1 / (1 - self.ab)

    def u(self, c):
        " Utility "
        return np.log(c)

    def u_prime(self, c):
        return 1 / c

    def f(self, k):
        " Deterministic part of production function.  "
        return k**self.alpha

    def f_prime(self, k):
        return self.alpha * k**(self.alpha - 1)

    def c_star(self, y):
        " True optimal policy.  "
        return (1 - self.alpha * self.beta) * y

    def v_star(self, y):
        " True value function. " # eqaution 12
        return self.c1 + self.c2 * (self.c3 - self.c4) + self.c4 * np.log(y) 
    
#################################################################################

lg = LogLinearOG()
# == Unpack parameters / functions for convenience == #
alpha, beta, mu, s = lg.alpha, lg.beta, lg.mu, lg.s
v_star = lg.v_star


grid_max = 4         # Largest grid point
grid_size = 200      # Number of grid points
shock_size = 250     # Number of shock draws in Monte Carlo integral

grid = np.linspace(1e-5, grid_max, grid_size)
shocks = np.exp(mu + s * np.random.randn(shock_size))

#################################################################################

import matplotlib.pyplot as plt
w, w_95, w_H = bellman_operator(v_star(grid),
                     grid,
                     beta,
                     np.log,
                     lambda k: k**alpha,
                     shocks)

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_ylim(-35, -24)
ax.plot(grid, w, lw=2, alpha=0.6, label=r'optimal policy approximaion')
ax.plot(grid, w_95, lw=2, alpha=0.6, label=r'$0.95y$')
ax.plot(grid, w_H, lw=2, alpha=0.6, label=r'$H(y)y*$')
ax.plot(grid, v_star(grid), lw=2, alpha=0.6, label='$v^*$')
ax.legend(loc='lower right')
plt.show()

#################################################################################

