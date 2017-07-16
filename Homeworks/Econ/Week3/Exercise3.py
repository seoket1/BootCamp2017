# Import packages
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit

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

dens = 2
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
def operating_profits(z_grid, kvec, alpha_l, alpha_k):
    for i in range(len(z_grid)):
        op[i] = ((1 - alpha_l) * ((alpha_l / w) ** (alpha_l / (1 - alpha_l))) * ((kvec ** alpha_k) ** (1 / (1 - alpha_l)))) * np.exp(z_grid[i]) ** (1/(1-alpha_l))
    return op
op = operating_profits(z_grid, kvec, alpha_l, alpha_k)

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
VFtol = 1e-2
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
    print(VFiter)
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
            
            
# Plot the solution
for i in range(len(z_grid)):
    plt.plot(kvec, VF[i, :], label='Value Function from z: %f' %(np.exp(z_grid[i])))
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
            


'''*************************'''
''' Initial Setting for PFI '''
'''*************************'''   

k0 = np.zeros((len(z_grid), len(kvec)))

for i in range (len(z_grid)):
    for j in range (len(kvec)):
        k0[i, j] = kvec[j]

c1 = np.zeros((len(z_grid), len(kvec), len(kvec)))
def c1_iter(psi, k0, delta, z_grid):
    for k in range(len(z_grid)):
        for i in range(len(kvec)):
            for j in range(len(kvec)):
                c1[k, i, j] = psi * ( (k0[k, j] - (1-delta)*k0[k, i])/k0[k, i] )
    return c1

c2 = np.zeros((len(z_grid), len(kvec), len(kvec)))
def c2_iter(psi, k0, delta, z_grid):
    for k in range(len(z_grid)):
        for i in range(len(kvec)):
            for j in range(len(kvec)):
                c2[k, i, j] = psi * ( (k0[k, j] - (1-delta)*k0[k, i])/k0[k, i] ) * (-k0[k, j]/k0[k, i]) + (psi/2) * ( (k0[k, j] - (1-delta)*k0[k, i])/k0[k, i] ) ** 2  
    return c2

pi2 = np.zeros((len(z_grid), len(kvec)))
def pi2_iter(alpha_k, w, alpha_l, z_grid, k0):
    for k in range(len(z_grid)):
        pi2[k] = alpha_k * (alpha_l/w)**(alpha_l/(1-alpha_l)) * np.exp(z_grid[k])**(1/(1-alpha_l)) * k0[k]**((alpha_k+alpha_l-1)/(1-alpha_l))
    return pi @ pi2
pi2_with_uncertainty = pi2_iter(alpha_k, w, alpha_l, z_grid, k0)



k_p = np.zeros((len(z_grid), sizek, sizek, sizek))
@jit
def FOC(pi2_with_uncertainty, delta, sizek, z_grid, c1, c2):
    for i in range(sizek):
        for j in range(sizek):
            for k in range(len(z_grid)):
                for t in range(sizek):
                    k_p[k, i, j, t] = betafirm * (pi2_with_uncertainty[k, j] + (1-delta) - c2[k, j, t]) - c1[k, i, j] - 1
    return k_p

PFtol = 1e-2
PFdist = 7.0
PFmaxiter = 3000
P = np.zeros((len(z_grid), sizek))  # initial guess at value function
Pmat = np.zeros((len(z_grid), sizek, sizek, sizek))  # initialize Vmat matrix

PFiter = 1

while PFdist > PFtol and PFiter < PFmaxiter:
    # New Calculation with new "K"
    c1 = c1_iter(psi, k0, delta, z_grid)
    c2 = c2_iter(psi, k0, delta, z_grid)
    pi2 = pi2_iter(alpha_k, w, alpha_l, z_grid, k0)
    foc = FOC(pi2_with_uncertainty, delta, sizek, z_grid, c1, c2)
    foc_final = abs(foc[:,:,:,0]).min(axis=2)
    k_PF = np.argmin(abs(foc[:,:,:,0]), axis=2)  # find the index of the optimal k'
    k1 = kvec[k_PF]
    PFdist = (np.absolute(k1 - k0)).max()  # check distance between Capital
    k0 = np.copy(k1)
    # function for this iteration and value function from past iteration
    print(PFiter)
    PFiter += 1

k_optK = kvec[PF]

# Restore Value Function from capital
@jit
def profits(z_grid, kvec, alpha_l, alpha_k):
    for i in range(len(z_grid)):
        for j in range(len(kvec)):
            op[i, j] = ((1 - alpha_l) * ((alpha_l / w) ** (alpha_l / (1 - alpha_l))) * ((kvec[i, j] ** alpha_k) ** (1 / (1 - alpha_l)))) * np.exp(z_grid[i]) ** (1/(1-alpha_l))
    return op

@jit
def Cash(e, op, kvec, delta, psi, sizek, z_grid):
    for i in range(sizek):
        for j in range(sizek):
            for k in range(len(z_grid)):
                e[k, i, j] = (op[k, i] - kvec[k, j] + ((1 - delta) * kvec[k, i]) - ((psi / 2) * ((kvec[k, j] - ((1 - delta) * kvec[k, i])) ** 2)/ kvec[k, i]))
    return e

opp = profits(z_grid, k_optK, alpha_l, alpha_k)
e = np.zeros((len(z_grid), sizek, sizek))
VVV = Cash(e, opp, k_optK, delta, psi, sizek, z_grid) *(1/(1-betafirm))
VVV = VVV.max(axis=2)

# Plot FOC
for i in range(len(z_grid)):
    plt.plot(kvec, foc_final[i, :], label='Value Function from z: %f' %(np.exp(z_grid[i])))
    plt.ylim(0,100)
plt.title('FOC. It hase to be zero')
plt.show()

# Plot the solution
for i in range(len(z_grid)):
    plt.plot(kvec, VVV[i, :], label='Value function from z: %f' %(np.exp(z_grid[i])))
plt.xlabel('Size of Capital Stock')
plt.ylabel('Value Function')
plt.title('Value Function - sthocastic firm w/ adjustment costs')   
plt.legend(loc=9, bbox_to_anchor=(1.2, 1))         
plt.show()       

# Plot optimal capital stock rule as a function of firm size
fig, ax = plt.subplots()
for i in range(len(z_grid)):
    ax.plot(kvec, k_optK[i, :], '--', label='Capital Next Period from z: %f' %(np.exp(z_grid[i])))
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



print("\n\n -- Comments --\n")
print("If I make policty function codes more efficient, it might be time-saving. But here, PFI takes longer time than VFI")
print("The reason why I think PFI might be faster is that iteration time of PFI is shorter than VFI's when I execute the codes.\n")
print("When I see the result, the functions made from both of them look very similar even though it is not exactly same.\n")
print("Because of processing time, I set weak tolerance and dense (1e-2 and 2). If I set those stonger, I expect that its shape will look more similar.")





