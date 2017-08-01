import numpy as np
import sympy as sy
from sympy.abc import x
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy import Function

# 2.2
''' For derivative '''
alpha = sy.symbols('alpha')
beta = sy.symbols('beta')
z0 = sy.symbols('z0')
z1 = sy.symbols('z1')
k0 = sy.symbols('k0')
k1 = sy.symbols('k1')
k2 = sy.symbols('k2')

Functions = beta*( alpha*sy.exp(z1)*sy.exp(k1)**(alpha-1)*(sy.exp(z0)*sy.exp(k0)**alpha - k1) / (sy.exp(z1)*sy.exp(k1)**alpha - sy.exp(k2)) )

fx = lambda z0, z1, k0, k1, k2, alpha, beta: Functions

print("\n", sy.diff(fx(z0, z1, k0, k1, k2, alpha, beta), k0))
print("\n", sy.diff(fx(z0, z1, k0, k1, k2, alpha, beta), k1))
print("\n", sy.diff(fx(z0, z1, k0, k1, k2, alpha, beta), k2))
print("\n", sy.diff(fx(z0, z1, k0, k1, k2, alpha, beta), z0))
print("\n", sy.diff(fx(z0, z1, k0, k1, k2, alpha, beta), z1))

gamma_k0 = sy.diff(fx(z0, z1, k0, k1, k2, alpha, beta), k0)
gamma_k1 = sy.diff(fx(z0, z1, k0, k1, k2, alpha, beta), k1)
gamma_k2 = sy.diff(fx(z0, z1, k0, k1, k2, alpha, beta), k2)
gamma_z0 = sy.diff(fx(z0, z1, k0, k1, k2, alpha, beta), z0)
gamma_z1 = sy.diff(fx(z0, z1, k0, k1, k2, alpha, beta), z1)
      
# exp(zbar) = kbar**(1-alpha)/alpha*beta

# Parameters
alpha = 0.35
beta = 0.98
rho = 0.95
sigma = 0.02

A = alpha * beta
kbar = A**(1/(1-alpha))

k0 = np.log(kbar)
k1 = np.log(kbar)
k2 = np.log(kbar)

z0 = kbar ** (1-alpha) / (alpha*beta)
z1 = kbar ** (1-alpha) / (alpha*beta)
gamma = A*np.exp(z0) * np.exp(kbar)**(alpha - 1)

sizek = 26
sizez = 26
k_grid = np.linspace(0.5*kbar, 1.5*kbar, sizek)
z_grid = np.linspace(-5*sigma, 5*sigma, sizez)


F = alpha*beta*(-k1 + np.exp(k0)**alpha*np.exp(z0))*np.exp(k1)**(alpha - 1)*np.exp(k2)*np.exp(z1)/(np.exp(k1)**alpha*np.exp(z1) - np.exp(k2))**2 / gamma
G = -alpha**2*beta*(-k1 + np.exp(k0)**alpha*np.exp(z0))*np.exp(k1)**alpha*np.exp(k1)**(alpha - 1)*np.exp(2*z1)/(np.exp(k1)**alpha*np.exp(z1) - np.exp(k2))**2 + alpha*beta*(alpha - 1)*(-k1 + np.exp(k0)**alpha*np.exp(z0))*np.exp(k1)**(alpha - 1)*np.exp(z1)/(np.exp(k1)**alpha*np.exp(z1) - np.exp(k2)) - alpha*beta*np.exp(k1)**(alpha - 1)*np.exp(z1)/(np.exp(k1)**alpha*np.exp(z1) - np.exp(k2)) / gamma
H = alpha*beta*(-k1 + np.exp(k0)**alpha*np.exp(z0))*np.exp(k1)**(alpha - 1)*np.exp(k2)*np.exp(z1)/(np.exp(k1)**alpha*np.exp(z1) - np.exp(k2))**2 / gamma
L = alpha*beta*(-k1 + np.exp(k0)**alpha*np.exp(z0))*np.exp(k1)**(alpha - 1)*np.exp(z1)/(np.exp(k1)**alpha*np.exp(z1) - np.exp(k2)) - alpha*beta*(-k1 + np.exp(k0)**alpha*np.exp(z0))*np.exp(k1)**alpha*np.exp(k1)**(alpha - 1)*np.exp(2*z1)/(np.exp(k1)**alpha*np.exp(z1) - np.exp(k2))**2  / gamma
M = alpha*beta*np.exp(k0)**alpha*np.exp(k1)**(alpha - 1)*np.exp(z0)*np.exp(z1)/(np.exp(k1)**alpha*np.exp(z1) - np.exp(k2))/ gamma
N = rho


'''
F = kbar * alpha*beta*k1**(alpha - 1)*(k0**alpha*np.exp(z0) - k1)*np.exp(z1)/(k1**alpha*np.exp(z1) - k2)**2 / gamma
G = kbar * -alpha**2*beta*k1**alpha*k1**(alpha - 1)*(k0**alpha*np.exp(z0) - k1)*np.exp(2*z1)/(k1*(k1**alpha*np.exp(z1) - k2)**2) - alpha*beta*k1**(alpha - 1)*np.exp(z1)/(k1**alpha*np.exp(z1) - k2) + alpha*beta*k1**(alpha - 1)*(alpha - 1)*(k0**alpha*np.exp(z0) - k1)*np.exp(z1)/(k1*(k1**alpha*np.exp(z1) - k2)) / gamma
H = kbar * alpha**2*beta*k0**alpha*k1**(alpha - 1)*np.exp(z0)*np.exp(z1)/(k0*(k1**alpha*np.exp(z1) - k2)) / gamma
L = -alpha*beta*k1**alpha*k1**(alpha - 1)*(k0**alpha*np.exp(z0) - k1)*np.exp(2*z1)/(k1**alpha*np.exp(z1) - k2)**2 + alpha*beta*k1**(alpha - 1)*(k0**alpha*np.exp(z0) - k1)*np.exp(z1)/(k1**alpha*np.exp(z1) - k2)  / gamma
M = alpha*beta*k0**alpha*k1**(alpha - 1)*np.exp(z0)*np.exp(z1)/(k1**alpha*np.exp(z1) - k2)/ gamma
N = rho


F = alpha * np.log(kbar) **(alpha -1) / (kbar **alpha - kbar)
G = - alpha * kbar **(alpha -1) * (alpha + kbar**(alpha -1)) / (kbar **alpha - kbar)
H = alpha**2 * kbar **(2*(alpha -1)) / (kbar **alpha - kbar)
L = - alpha * kbar **(2*alpha -1) / (kbar **alpha - kbar)
M = alpha**2 * kbar **(2*(alpha -1)) / (kbar **alpha - kbar)
N = rho
'''

P = (-G + np.sqrt(G**2 - 4*F*H)) / (2*F)
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
ax.view_init(elev=20., azim=-20)  # to rotate plot for better view
ax.set_xlabel(r'z_grid')
ax.set_ylabel(r'k')
ax.set_zlabel(r'k+1')