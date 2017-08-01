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
FF = F_1 
G = F_2
H = F_3 
L = F_4 
M = F_5 
N = rho

P = (-G - np.sqrt(G**2 - 4*FF*H)) / (2*FF) 
Q = -(L*N + M) / (FF*N + FF*P + G)  

Hx = np.copy(P)
Hz = np.copy(Q)
Hv = 0

# Setting Gamma
GammaA = np.array([F_1, F_2, F_3, F_4, F_5])
GammaAA= np.array([[F_1_1, F_1_2, F_1_3, F_1_4, F_1_5],\
                    [F_2_1, F_2_2, F_2_3, F_2_4, F_2_5],\
                    [F_3_1, F_3_2, F_3_3, F_3_4, F_3_5],\
                    [F_4_1, F_4_2, F_4_3, F_4_4, F_4_5],\
                    [F_5_1, F_5_2, F_5_3, F_5_4, F_5_5]])

def quad(theta_init):
    Hxx, Hxz, Hzz, Hvv = theta_init
    
    EFss = ([[Hxx*Hx*Hx + Hx*Hxx, Hxx*Hx*Hz + Hx * Hxz + Hxz*Hx*N, Hxx*Hx*Hv],
          [Hxx, Hxz, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [Hxx*Hx*Hz + Hx*Hxz + Hxz*Hx*N, Hxz*N*Hz + Hx *Hzz + Hzz*N*N, Hxz*N*Hv],
          [Hxz, Hzz, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [Hxx*Hx*Hv, Hxz*N*Hv, Hx*Hvv + Hvv],
          [0,0,Hvv],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],])
    
    EFs =  np.array([[Hx*Hx, Hx*Hz + Hz*N, Hx*Hv + Hv],
                     [Hx, Hz, Hv],
                     [1, 0, 0],
                     [0, N, 0],
                     [0, 1, 0]])
    
    ELambda = ((np.kron(EFs, np.eye(nx + ny))).T@GammaAA) @ EFs
    Deltass = ELambda + np.kron(np.eye(ns), GammaA) @ EFss
    
    a = Deltass[0, :]
    b = Deltass[1, :]
    c = Deltass[2, :]
    
    return np.hstack((a,b,c))

theta_init = np.array([-1, 1, 1, 0.1])
theta_opt = opt.root(quad, theta_init, method = 'lm').x

Hxx_o, Hxz_o, Hzz_o, Hvv_o = theta_opt

opt_K_purtur = np.zeros((sizek, sizez))
for i in range(sizez):
    opt_K_purtur[:, i] = kbar + Hx* (k_grid - kbar) +\
                Hz*z_grid[i] + 0.5 * Hxx_o * (k_grid - kbar)**2\
                + 0.5*Hzz_o * z_grid[i]**2 + 0.5*2*Hxz_o*(k_grid-kbar)*z_grid[i]

## plot the SD
# import packages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Stationary distribution in 3D
kmat, zmat = np.meshgrid(k_grid, z_grid)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(zmat, kmat, opt_K_purtur, rstride=1, cstride=1, cmap=cm.Blues,
                linewidth=0, antialiased=False)
ax.view_init(elev=20., azim=30)  # to rotate plot for better view
ax.set_xlabel(r'z_grid')
ax.set_ylabel(r'k')
ax.set_zlabel(r'k+1')








