# Import packages
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit
from scipy.optimize import root
from scipy.optimize import fixed_point
# import packages for stochastic variables
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



'''######################'''
'''For stochastic parts'''
'''######################'''
# set our parameters
rho = 0.7605
mu = 0.0
sigma_eps = 0.213
N = 9  # number of grid points (will have one more cut-off point than this)

# draw our shocks
num_draws = 100000 # number of shocks to draw
eps = np.random.normal(0.0, sigma_eps, size=(num_draws))

# Compute z
z = np.empty(num_draws)
z[0] = 0.0 + eps[0]
for i in range(1, num_draws):
    z[i] = rho * z[i - 1] + (1 - rho) * mu + eps[i]
    
# theory says:
sigma_z = sigma_eps / ((1 - rho ** 2) ** (1 / 2))
print('Theoretical sigma_z = ', sigma_z)

# from our simulation:
sigma_z_simul = z.std()
print('Simulated sigma_z = ', sigma_z_simul)

# import packages
from scipy.stats import norm

# Compute cut-off values
N = 9  # number of grid points (will have one more cut-off point than this)
z_cutoffs = (sigma_z * norm.ppf(np.arange(N + 1) / N)) + mu
print('Cut-off values = ', z_cutoffs)

# compute grid points for z
z_grid = ((N * sigma_z * (norm.pdf((z_cutoffs[:-1] - mu) / sigma_z)
                              - norm.pdf((z_cutoffs[1:] - mu) / sigma_z)))
              + mu)
print('Grid points = ', z_grid)

# import packages
import scipy.integrate as integrate

# define function that we will integrate
@jit      
def integrand(x, sigma_z, sigma_eps, rho, mu, z_j, z_jp1):
    val = (np.exp((-1 * ((x - mu) ** 2)) / (2 * (sigma_z ** 2)))
            * (norm.cdf((z_jp1 - (mu * (1 - rho)) - (rho * x)) / sigma_eps)
               - norm.cdf((z_j - (mu * (1 - rho)) - (rho * x)) / sigma_eps)))
    return val

# compute transition probabilities
pi = np.empty((N, N))

for i in range(N):
    for j in range(N):
        results = integrate.quad(integrand, z_cutoffs[i], z_cutoffs[i + 1],
                                 args = (sigma_z, sigma_eps, rho, mu,
                                         z_cutoffs[j], z_cutoffs[j + 1]))
        pi[i,j] = (N / np.sqrt(2 * np.pi * sigma_z ** 2)) * results[0]
        
# print('Transition matrix = ', pi)
# print('pi sums = ', pi.sum(axis=0), pi.sum(axis=1))

# Simulate the Markov process - will make this a function so can call later
@jit
def sim_markov(z_grid, pi, num_draws):
    # draw some random numbers on [0, 1]
    u = np.random.uniform(size=num_draws)

    # Do simulations
    z_discrete = np.empty(num_draws)  # this will be a vector of values 
    # we land on in the discretized grid for z
    N = z_grid.shape[0]
    oldind = int(np.ceil((N - 1) / 2)) # set initial value to median of grid
    z_discrete[0] = z_grid[oldind]  
    for i in range(1, num_draws):
        sum_p = 0
        ind = 0
        while sum_p < u[i]:
            sum_p = sum_p + pi[ind, oldind]
#             print('inds =  ', ind, oldind)
            ind += 1
        if ind > 0:
            ind -= 1
        z_discrete[i] = z_grid[ind]
        oldind = ind
                            
    return z_discrete

# Call simulation function to get simulated values
z_discrete = sim_markov(z_grid, np.transpose(pi), num_draws)
                            


'''######################'''
'''For determinstic parts'''
'''######################'''
# specify parameters
alpha_k = 0.29715
alpha_l = 0.65
delta = 0.154
psi = 1.08
w = 0.7
r= 0.04
z = 1

betafirm = (1 / (1 + r))

dens = 5
# put in bounds here for the capital stock space
kstar = ((((1 / betafirm - 1 + delta) * ((w / alpha_l) **
                                         (alpha_l / (1 - alpha_l)))) /
         (alpha_k * (z ** (1 / (1 - alpha_l))))) **
         ((1 - alpha_l) / (alpha_k + alpha_l - 1)))
kbar = 2*kstar
lb_k = 0.001
ub_k = kbar
krat = np.log(lb_k / ub_k)
numb = np.ceil(krat / np.log(1 - delta))
K = np.zeros(int(numb * dens))

# we'll create in a way where we pin down the upper bound - since
# the distance will be small near the lower bound, we'll miss that by little
for j in range(int(numb * dens)):
    K[j] = ub_k * (1 - delta) ** (j / dens)
kvec = K[::-1]
sizek = kvec.shape[0]

'''************************'''
'''to input adjustment cost'''
'''************************'''
# operating profits, op
op = np.zeros((len(z_grid), len(kvec)))

@jit
def operating_profits(z_grid, kvec, alpha_l, alpha_k, w):
    for i in range(len(z_grid)):
        op[i] = ((1 - alpha_l) * ((alpha_l / w) ** (alpha_l / (1 - alpha_l))) * ((kvec ** alpha_k) ** (1 / (1 - alpha_l)))) * np.exp(z_grid[i]) ** (1/(1-alpha_l))
    return op
op = operating_profits(z_grid, kvec, alpha_l, alpha_k, w)

# firm cash flow, e
e = np.zeros((len(z_grid), sizek, sizek))

@jit
def Cash_flow(e, op, kvec, delta, psi, sizek, z_grid):
    for i in range(sizek):
        for j in range(sizek):
            for k in range(len(z_grid)):
                e[k, i, j] = (op[k, i] - kvec[j] + ((1 - delta) * kvec[i]) - ((psi / 2) * ((kvec[j] - ((1 - delta) * kvec[i])) ** 2)/ kvec[i]))
    return e

e = Cash_flow(e, op, kvec, delta, psi, sizek, z_grid)


'''*************************'''
''' Initial Setting for VFI '''
'''*************************'''      
VFtol = 1e-6
VFdist = 7.0
VFmaxiter = 3000
V = np.zeros((len(z_grid), sizek))  # initial guess at value function
Vmat = np.zeros((len(z_grid), sizek, sizek))  # initialize Vmat matrix
Vstore = np.zeros((len(z_grid),sizek, VFmaxiter))  # initialize Vstore array
VFiter = 1
start_time = time.clock()

@jit
def vmat(sizek, z_grid, e, betafirm, V, Vmat):
    for i in range(sizek):  # loop over k
        for j in range(sizek):  # loop over k'
            for k in range(len(z_grid)):  # loop over z
                Vmat[k, i, j] = e[k, i, j] + betafirm * V[k, j]
    return Vmat

                
while VFdist > VFtol and VFiter < VFmaxiter:
    TV = np.copy(V)
    V = np.copy(pi @ V)
    
    Vmat = vmat(sizek, z_grid, e, betafirm, V, Vmat)
    Vstore[:,:, VFiter] = np.copy(V.reshape(len(z_grid), sizek,))  # store value function at each
    # iteration for graphing later
    V = Vmat.max(axis=2)  # apply max operator to Vmat (to get V(k))
    PF = np.argmax(Vmat, axis=2)  # find the index of the optimal k'
    VFdist = (np.absolute(V - TV)).max()  # check distance between value
    # function for this iteration and value function from past iteration
    VFiter += 1
    
VFI_time = time.clock() - start_time
if VFiter < VFmaxiter:
    print('Value function converged after this many iterations:', VFiter)
else:
    print('Value function did not converge')
print('VFI took ', VFI_time, ' seconds to solve')

VF = V  # solution to the functional equation

'''
------------------------------------------------------------------------
Find optimal capital and investment policy functions
------------------------------------------------------------------------
optK = (len(z_grid), sizek,) vector, optimal choice of k' for each k
optI = (len(z_grid), sizek,) vector, optimal choice of investment for each k
------------------------------------------------------------------------
'''


optK = kvec[PF]
optI = optK - (1 - delta) * kvec   
            
'''            
# Plot the solution
for i in range(len(z_grid)):
    plt.plot(kvec, VF[i, :], label='Value function from z: %f' %(np.exp(z_grid[i])))
plt.xlabel('Size of Capital Stock')
plt.ylabel('Value Function')
plt.title('Value Function - sthocastic firm w/ adjustment costs')   
plt.legend(loc=9, bbox_to_anchor=(1.2, 1))         
plt.show()       
            
# Plot optimal capital stock rule as a function of firm size
fig, ax = plt.subplots()
for i in range(len(z_grid)):
    ax.plot(kvec, optK[i, :], '--', label='Capital Next Period from z: %f' %(np.exp(z_grid[i])))
ax.plot(kvec, kvec, 'k:', linewidth=5, label='45 degree line')
# Now add the legend with some customizations.
legend = ax.legend(loc='upper left', shadow=True)
# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('Size of Capital Stock')
plt.ylabel('Optimal Choice of Capital Next Period')
plt.title('Policy Function, Next Period Capoital - stochastic firm w/ ' +
          'adjustment costs')
plt.legend(loc=9, bbox_to_anchor=(1, 0.5))
plt.show()
'''        
    
'''****************************'''
''' Counsumer Side Calculation '''
'''****************************'''     

'''------------------------------------------------------------------------
Compute the stationary distribution of firms over (k, z)
------------------------------------------------------------------------
SDtol     = tolerance required for convergence of SD
SDdist    = distance between last two distributions
SDiter    = current iteration
SDmaxiter = maximium iterations allowed to find stationary distribution
Gamma     = stationary distribution
HGamma    = operated on stationary distribution
------------------------------------------------------------------------
'''

# Given Values from Problem 4
h = 6.616


def Gamma(PF, sizez, Pi):
    Gamma = np.ones((sizez, sizek)) * (1 / (sizek * sizez))
    SDtol = 1e-12
    SDdist = 7
    SDiter = 0
    SDmaxiter = 1000
    while SDdist > SDtol and SDmaxiter > SDiter:
        HGamma = np.zeros((sizez, sizek))
        for i in range(sizez):  # z
            for j in range(sizek):  # k
                for m in range(sizez):  # z'
                    HGamma[m, PF[i, j]] = \
                        HGamma[m, PF[i, j]] + Pi[i, m] * Gamma[i, j]
        SDdist = (np.absolute(HGamma - Gamma)).max()
        Gamma = HGamma
        SDiter += 1
    print("found gamma")
    return Gamma

def get_Y(z_grid, sizek, optK, alpha_k, alpha_l, gamma):
    labor_demand_before = np.zeros((len(z_grid), sizek))
    for i in range(len(z_grid)):
        for j in range(sizek):
            labor_demand_before[i, j] = ((alpha_l / w) ** (1 / (1 - alpha_l))) * ((optK[i,j] ** alpha_k) ** (1 / (1 - alpha_l))) * np.exp(z_grid[i]) ** (1/(1-alpha_l))
    Y_before = np.zeros((len(z_grid), sizek))
    for i in range(len(z_grid)):
        for j in range(sizek):
            Y_before[i,j] = z_grid[i] * (optK[i,j]) ** alpha_k * (labor_demand_before[i,j]) **alpha_l
    Y_bar = np.sum(Y_before*gamma)
    return Y_bar    

def get_labor_demand(z_grid, sizek, alpha_l, w, optK, alpha_k, gamma):
    labor_demand_before = np.zeros((len(z_grid), sizek))
    for i in range(len(z_grid)):
        for j in range(sizek):
            labor_demand_before[i, j] = ((alpha_l / w) ** (1 / (1 - alpha_l))) * ((optK[i,j] ** alpha_k) ** (1 / (1 - alpha_l))) * np.exp(z_grid[i]) ** (1/(1-alpha_l))
    labor_demand_bar = np.sum(labor_demand_before * gamma)
    return labor_demand_bar
    
def get_labor_supply(Y_bar, I_bar, cost_bar, consumption, h):
    consumption = Y_bar - I_bar - cost_bar
    labor_supply = w/(consumption*h)
    return labor_supply

def get_investment(optI, gamma):
    I_bar =  np.sum(optI*gamma)
    return I_bar

def get_adjustment_cost(gamma, z_grid, sizek, psi, optK, delta):
    cost_bar_before = np.zeros((len(z_grid), sizek))
    for i in range(len(z_grid)):
        for j in range(sizek):
            cost_bar_before[i, j] =  ((psi / 2) * ((optK[i, j] - ((1 - delta) * optK[i, j])) ** 2)/ optK[i, j])
    cost_bar = np.sum(cost_bar_before * gamma)
    return cost_bar    

def get_distance(w):
    print("\nIteration starts for wage")
########################################################################################
########################################################################################
########################################################################################
    op = operating_profits(z_grid, kvec, alpha_l, alpha_k, w)
    e = np.zeros((len(z_grid), sizek, sizek))
    e = Cash_flow(e, op, kvec, delta, psi, sizek, z_grid)
    
    VFtol = 1e-6
    VFdist = 7.0
    VFmaxiter = 3000
    V = np.zeros((len(z_grid), sizek))  # initial guess at value function
    Vmat = np.zeros((len(z_grid), sizek, sizek))  # initialize Vmat matrix
    Vstore = np.zeros((len(z_grid),sizek, VFmaxiter))  # initialize Vstore array
    VFiter = 1
    start_time = time.clock()
    @jit
    def vmat(sizek, z_grid, e, betafirm, V, Vmat):
        for i in range(sizek):  # loop over k
            for j in range(sizek):  # loop over k'
                for k in range(len(z_grid)):  # loop over z
                    Vmat[k, i, j] = e[k, i, j] + betafirm * V[k, j]
        return Vmat
         
    while VFdist > VFtol and VFiter < VFmaxiter:
        TV = np.copy(V)
        V = np.copy(pi @ V)
        
        Vmat = vmat(sizek, z_grid, e, betafirm, V, Vmat)
        Vstore[:,:, VFiter] = np.copy(V.reshape(len(z_grid), sizek,))  # store value function at each
        # iteration for graphing later
        V = Vmat.max(axis=2)  # apply max operator to Vmat (to get V(k))
        PF = np.argmax(Vmat, axis=2)  # find the index of the optimal k'
        VFdist = (np.absolute(V - TV)).max()  # check distance between value
        # function for this iteration and value function from past iteration
        VFiter += 1
    
    VFI_time = time.clock() - start_time
    if VFiter < VFmaxiter:
        print('Value function converged after this many iterations:', VFiter)
    else:
        print('Value function did not converge')

    print('VFI took ', VFI_time, ' seconds to solve')
    
    optK = kvec[PF]
    optI = optK - (1 - delta) * kvec   
########################################################################################
########################################################################################
########################################################################################

    # For market clearing condition....
    gamma = Gamma(PF, len(z_grid), pi)    
    labor_demand = get_labor_demand(z_grid, sizek, alpha_l, w, optK, alpha_k, gamma)
    I_bar = get_investment(optI, gamma)
    cost_bar = get_adjustment_cost(gamma, z_grid, sizek, psi, optK, delta)
    Y_bar = get_Y(z_grid, sizek, optK, alpha_k, alpha_l, gamma)
    consumption = Y_bar - I_bar - cost_bar
    labor_supply = get_labor_supply(Y_bar, I_bar, cost_bar, consumption, h) 
    distance = abs(labor_demand - labor_supply)
    print("distance =", distance)
    print("w =", w)
    print("Labor demand =", labor_demand)
    print("Labor supply =", labor_supply)

    return distance


# To find Wage_Bar!!!
Wage_Bar = root(get_distance, 1.0468, tol=1e-4)
print(Wage_Bar)

''' converged wage is 1.04695061"
distance = 0.00245998272636
w = [ 1.04695061]
Labor demand = 0.482755926825
Labor supply = 0.485215909551
'''

New_Wage = Wage_Bar["x"]
## plot the SD

# import packages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm




'''****************************************************************************'''
'''****************************************************************************'''
'''****************************************************************************'''
'''***********    In order to plot the Distribution     ***********************'''
'''****************************************************************************'''
'''****************************************************************************'''
'''****************************************************************************'''

########################################################################################
########################################################################################
########################################################################################
op = operating_profits(z_grid, kvec, alpha_l, alpha_k, New_Wage)
e = np.zeros((len(z_grid), sizek, sizek))
e = Cash_flow(e, op, kvec, delta, psi, sizek, z_grid)

VFtol = 1e-6
VFdist = 7.0
VFmaxiter = 3000
V = np.zeros((len(z_grid), sizek))  # initial guess at value function
Vmat = np.zeros((len(z_grid), sizek, sizek))  # initialize Vmat matrix
Vstore = np.zeros((len(z_grid),sizek, VFmaxiter))  # initialize Vstore array
VFiter = 1
start_time = time.clock()

@jit
def vmat(sizek, z_grid, e, betafirm, V, Vmat):
    for i in range(sizek):  # loop over k
        for j in range(sizek):  # loop over k'
            for k in range(len(z_grid)):  # loop over z
                Vmat[k, i, j] = e[k, i, j] + betafirm * V[k, j]
    return Vmat

                
while VFdist > VFtol and VFiter < VFmaxiter:
    TV = np.copy(V)
    V = np.copy(pi @ V)
    
    Vmat = vmat(sizek, z_grid, e, betafirm, V, Vmat)
    Vstore[:,:, VFiter] = np.copy(V.reshape(len(z_grid), sizek,))  # store value function at each
    # iteration for graphing later
    V = Vmat.max(axis=2)  # apply max operator to Vmat (to get V(k))
    PF = np.argmax(Vmat, axis=2)  # find the index of the optimal k'
    VFdist = (np.absolute(V - TV)).max()  # check distance between value
    # function for this iteration and value function from past iteration
    VFiter += 1
    
VFI_time = time.clock() - start_time
if VFiter < VFmaxiter:
    print('Value function converged after this many iterations:', VFiter)
else:
    print('Value function did not converge')
print('VFI took ', VFI_time, ' seconds to solve')

VF = V  # solution to the functional equation
    
########################################################################################
########################################################################################
########################################################################################
gamma = Gamma(PF, len(z_grid), pi) 

    
# Plot the stationary distribution over k
fig, ax = plt.subplots()
ax.plot(kvec, gamma.sum(axis=0))
plt.xlabel('Size of Capital Stock')
plt.ylabel('Density')
plt.title('Stationary Distribution over Capital')
plt.show()

# Plot the stationary distribution
fig, ax = plt.subplots()
ax.plot(z_grid, gamma.sum(axis=1))
plt.xlabel('Log Productivity')
plt.ylabel('Density')
plt.title('Stationary Distribution over Productivity')

# Stationary distribution in 3D
zmat, kmat = np.meshgrid(kvec, z_grid)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmat, zmat, gamma, rstride=1, cstride=1, cmap=cm.Blues,
                linewidth=0, antialiased=False)
ax.view_init(elev=20., azim=20)  # to rotate plot for better view
ax.set_xlabel(r'Log Productivity')
ax.set_ylabel(r'Capital Stock')
ax.set_zlabel(r'Density')


'''
-- Comments --

I wrote some comments on the pdf file.
'''

