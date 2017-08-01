import numpy as np
import scipy.optimize as opt
from numba import jit

# 1.8
# Parameters
alpha = 0.35
beta = 0.98
rho = 0.95
sigma = 0.02

A = alpha * beta
kbar = A**(1/(1-alpha))

sizek = 26
sizez = 26
k_grid = np.linspace(0.5*kbar, 1.5*kbar, sizek)
z_grid = np.linspace(-5*sigma, 5*sigma, sizez)

# transition matrix
def rouwen(rho, mu, step, num):
    '''
    Adapted from Lu Zhang and Karen Kopecky. Python by Ben Tengelsen.
    Construct transition probability matrix for discretizing an AR(1)
    process. This procedure is from Rouwenhorst (1995), which works
    well for very persistent processes.

    INPUTS:
    rho  - persistence (close to one)
    mu   - mean and the middle point of the discrete state space
    step - step size of the even-spaced grid
    num  - number of grid points on the discretized process

    OUTPUT:
    dscSp  - discrete state space (num by 1 vector)
    transP - transition probability matrix over the grid
    '''

    # discrete state space
    dscSp = np.linspace(mu -(num-1)/2*step, mu +(num-1)/2*step, num).T

    # transition probability matrix
    q = p = (rho + 1)/2.
    transP = np.array([[p**2, p*(1-q), (1-q)**2], \
                    [2*p*(1-p), p*q+(1-p)*(1-q), 2*q*(1-q)], \
                    [(1-p)**2, (1-p)*q, q**2]]).T


    while transP.shape[0] <= num - 1:

        # see Rouwenhorst 1995
        len_P = transP.shape[0]
        transP = p * np.vstack((np.hstack((transP, np.zeros((len_P, 1)))), np.zeros((1, len_P+1)))) \
                + (1 - p) * np.vstack((np.hstack((np.zeros((len_P, 1)), transP)), np.zeros((1, len_P+1)))) \
                + (1 - q) * np.vstack((np.zeros((1, len_P+1)), np.hstack((transP, np.zeros((len_P, 1)))))) \
                + q * np.vstack((np.zeros((1, len_P+1)), np.hstack((np.zeros((len_P, 1)), transP))))

        transP[1:-1] /= 2.


    # ensure columns sum to 1
    if np.max(np.abs(np.sum(transP, axis=1) - np.ones(transP.shape))) >= 1e-12:
        print('Problem in rouwen routine!')
        return None
    else:
        return transP.T, dscSp

pi, z_grid_rowen =  rouwen(rho, 0, sigma/25*10, sizez)
pi = pi.T

u = np.zeros((sizez, sizek, sizek))

@jit
def utility(z_grid, k_grid, alpha):
    for i in range(sizek): # k
        for j in range(sizek): # k+1
            for k in range(sizez): # various z_grid
                u[k, i, j] = np.log(np.exp(z_grid[k])*k_grid[i]**alpha - k_grid[j])
    return u

@jit
def vmat(sizez, sizek, u, beta, V, Vmat):
    for i in range(sizek):  # loop over k
        for j in range(sizek):  # loop over k'
            for k in range(sizez):  # loop over z
                Vmat[k, i, j] = u[k, i, j] + beta * V[k, j]
    return Vmat

u = utility(z_grid, k_grid, alpha)

VFtol = 1e-6
VFdist = 7.0
VFmaxiter = 3000
V = np.zeros((sizez, sizek))  # initial guess at value function
Vmat = np.zeros((sizez, sizek, sizek))  # initialize Vmat matrix
Vstore = np.zeros((sizez, sizek, VFmaxiter))  # initialize Vstore array
VFiter = 1


while VFdist > VFtol and VFiter < VFmaxiter:
    TV = np.copy(V)
    V = np.copy(pi @ V)
    
    Vmat = vmat(sizez, sizek, u, beta, V, Vmat)
    Vstore[:,:, VFiter] = np.copy(V.reshape(sizez, sizek,))  # store value function at each
    # iteration for graphing later
    V = Vmat.max(axis=2)  # apply max operator to Vmat (to get V(k))
    PF = np.argmax(Vmat, axis=2)  # find the index of the optimal k'
    VFdist = (np.absolute(V - TV)).max()  # check distance between value
    # function for this iteration and value function from past iteration
    VFiter += 1
    
if VFiter < VFmaxiter:
    print('Value function converged after this many iterations:', VFiter)
else:
    print('Value function did not converge')
    
VF = V  # solution to the functional equation 
    
'''
------------------------------------------------------------------------
Find optimal capital and investment policy functions
------------------------------------------------------------------------
optK = (sizez, sizek,) vector, optimal choice of k' for each k
optI = (sizez, sizek,) vector, optimal choice of investment for each k
------------------------------------------------------------------------
'''
optK = k_grid[PF]
#optI = optK - (1 - delta) * k_grid
            
# Plot the solution
for i in range(sizez):
    plt.plot(k_grid, VF[i, :], label='Investment rate from z: %f' %(z_grid[i]))
plt.xlabel('Size of Capital Stock')
plt.ylabel('Value Function')
plt.title('Value Function')   
plt.legend(loc=9, bbox_to_anchor=(1.2, 1))         
plt.show()       

# Plot optimal capital stock rule as a function of firm size
fig, ax = plt.subplots()
for i in range(len(z_grid)):
    ax.plot(k_grid, optK[i, :], '--', label='Capital Next Period from z: %f' %(z_grid[i]))
ax.plot(k_grid, k_grid, 'k:', linewidth=5, label='45 degree line')
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
plt.title('Policy Function, Next Period Capoital')
plt.legend(loc=9, bbox_to_anchor=(1.2, 1))
plt.show()
    
## plot the SD
# import packages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Stationary distribution in 3D
kmat, zmat = np.meshgrid(k_grid, z_grid)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(zmat, kmat, optK, rstride=1, cstride=1, cmap=cm.Blues,
                linewidth=0, antialiased=False)
ax.view_init(elev=20., azim=30)  # to rotate plot for better view
ax.set_xlabel(r'z_grid')
ax.set_ylabel(r'k')
ax.set_zlabel(r'k+1')