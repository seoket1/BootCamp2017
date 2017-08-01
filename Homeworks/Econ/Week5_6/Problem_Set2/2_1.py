import numpy as np
import sympy as sy
import scipy.optimize as opt
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy import Function
from matplotlib import pyplot as plt

# Parameters
alpha = 0.35
beta = 0.98
rho = 0.95
sigma = 0.02

A = alpha * beta
kbar = A**(1/(1-alpha))
zbar = 0

sizek = 26
sizez = 26
k_grid = np.linspace(0.5*kbar, 1.5*kbar, sizek)
z_grid = np.linspace(-5*sigma, 5*sigma, sizez)

# Setting Cardinality of each variable.
nx = 1
ny = 0
nz = 1

na = 3*nx + 2*ny + 2*nz
ns = nx + nz + 1

# Setting Gamma_A and Gamma_AA
F = lambda kpp, kp, k, zp, z: beta*(alpha*sy.exp(zp)*kp**(alpha-1)) * (sy.exp(z)*k**alpha - kp) / (sy.exp(zp)*kp**alpha - kpp) 

k = sy.symbols('k')
kp = sy.symbols('kp')
kpp = sy.symbols('kpp')
z = sy.symbols('z')
zp = sy.symbols('zp')

F_1 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), kpp))(kbar, kbar, kbar, zbar, zbar)
F_2 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), kp))(kbar, kbar, kbar, zbar, zbar)
F_3 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), k))(kbar, kbar, kbar, zbar, zbar)
F_4 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), zp))(kbar, kbar, kbar, zbar, zbar)
F_5 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), z))(kbar, kbar, kbar, zbar, zbar)

F_1_1 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), kpp, kpp))(kbar, kbar, kbar, zbar, zbar)
F_1_2 = F_2_1 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), kpp, kp))(kbar, kbar, kbar, zbar, zbar)
F_1_3 = F_3_1 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), kpp, k))(kbar, kbar, kbar, zbar, zbar)
F_1_4 = F_4_1 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), kpp, zp))(kbar, kbar, kbar, zbar, zbar)
F_1_5 = F_5_1 =  lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), kpp, z))(kbar, kbar, kbar, zbar, zbar)

F_2_2 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), kp, kp))(kbar, kbar, kbar, zbar, zbar)
F_2_3 = F_3_2 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), kp, k))(kbar, kbar, kbar, zbar, zbar)
F_2_4 = F_4_2 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), kp, zp))(kbar, kbar, kbar, zbar, zbar)
F_2_5 = F_5_2 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), kp, z))(kbar, kbar, kbar, zbar, zbar)

F_3_3 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), k, k))(kbar, kbar, kbar, zbar, zbar)
F_3_4 = F_4_3 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), k, zp))(kbar, kbar, kbar, zbar, zbar)
F_3_5 = F_5_3 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), k, z))(kbar, kbar, kbar, zbar, zbar)

F_4_4 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), zp, zp))(kbar, kbar, kbar, zbar, zbar)
F_4_5 = F_5_4 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), zp, z))(kbar, kbar, kbar, zbar, zbar)

F_5_5 = lambdify((kpp, kp, k, zp, z), sy.diff(F(kpp, kp, k, zp, z), z, z))(kbar, kbar, kbar, zbar, zbar)

# setting Hx, Hx, Hv
F = F_1 
G = F_2
H = F_3 
L = F_4 
M = F_5 
N = rho

P = (-G - np.sqrt(G**2 - 4*F*H)) / (2*F)
Q = -(L*N + M) / (F*N + F*P + G)

optK = np.zeros((sizek, sizez))
for i in range(sizez):
    optK[:,i] = kbar + P*(k_grid - kbar) + Q*z_grid[i]


## plot the SD
# import packages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Stationary distribution in 3D
kmat, zmat = np.meshgrid(k_grid, z_grid)
optK_3D = kbar + P*(kmat - kbar) + Q*zmat

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(zmat, kmat, optK, rstride=1, cstride=1, cmap=cm.Blues,
                linewidth=0, antialiased=False)
ax.view_init(elev=20., azim=10)  # to rotate plot for better view
ax.set_xlabel(r'z_grid')
ax.set_ylabel(r'k')
ax.set_zlabel(r'k+1')