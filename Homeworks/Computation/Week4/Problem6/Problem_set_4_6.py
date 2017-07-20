import numpy as np
from scipy import linalg as la
from scipy import optimize
from scipy.misc import derivative
from matplotlib import pyplot as plt
import cmath
import math
from scipy.sparse import dok_matrix
import sympy as sy
from matplotlib import pyplot as plt
from autograd import grad
import autograd.numpy as anp
from autograd import jacobian

#LINE Problem 1

def L1(x):
    return np.exp(x) - 4*x

def golden(f, a, b, niter):
    rho = (1/2)*(3-np.sqrt(5))
    a = a + rho*(b-a)
    b = a + (1-rho)*(b-a)
    n = 0
    midpoint = (1/2)*(a + b)

    while (abs(b-a) > 0.001) & (n < niter):
        
        if f(a) >= f(b):
            a = a + rho*(b-a)
            midpoint = (1/2)*(a + b)
        elif f(a) <= f(b):
            b = a + (1-rho)*(b-a)
            midpoint = (1/2)*(a + b)
            
        n = n + 1
        
    return n, midpoint 

print(golden(L1, 0, 3, 1000))

#LINE Problem 2

def bisect(f, a, b, niter):
    x = (b+a)/2
    n = 0
    
    while(abs(b-a) > 0.001) & (n < niter):
        fprime = derivative(f, x)
        if fprime >= 0:
            b = x
        elif fprime <= 0:
            a = x
        x = (b+a)/2
        n = n + 1
        
    return n, x

print(bisect(L1, 0, 3, 1000))

#LINE Problem 3

def L2(x):
    return (x**2) + np.sin(5*x)

def newton(fprime, fprimeprime, initial):
    xold = initial
    xnew = xold - fprime(xold)/fprimeprime(xold)
    n = 1
    diff = 5
    while(diff > 1e-5):
        xold = np.copy(xnew)
        xnew = xold - (fprime(xold)/fprimeprime(xold))
        diff = np.linalg.norm(xnew - xold)/np.linalg.norm(xold)
        n = n + 1
    return n, xnew

def L2prime(x):
    return 2*x + 5*np.cos(5*x)

def L2primeprime(x):
    return 2 - 25*np.sin(5*x)

print(newton(L2prime, L2primeprime, 0))

#LINE Problem 4

def L3(x):
    return (x**2) + np.sin(x) + np.sin(10*x)

def L3prime(x):
    return 2*x + np.cos(x) + 10*np.cos(10*x)

def secant(fprime, x1, x2):
    xold = x1
    xoldold = x2
    fppapprox = (fprime(xold) - fprime(xoldold))/(xold - xoldold)
    xnew = xold - fprime(xold)/fppapprox
    n = 1
    diff = 5
    while(diff > 1e-5):
        xoldold = np.copy(xold)
        xold = np.copy(xnew)
        fppapprox = (fprime(xold) - fprime(xoldold))/(xold - xoldold)
        xnew = xold - fprime(xold)/fppapprox
        diff = abs(xnew - xold)/abs(xold)
        n = n + 1
    return n, xnew

print(secant(L3prime, 0, -1))
print(secant(L3prime, -0.15, -0.2))
print(secant(L3prime, -0.15, -1))

#LINE Problem 5

def backtrack(f, jacob, x, p):
    alpha = 1
    c = 0.5
    rho = 0.5
    
    value = jacob(x).T @ p
    while (f(x + alpha*p) > f(x) + c*alpha*value):
        alpha = rho*alpha
    
    return alpha

from scipy.optimize import line_search

def objective(x):
    return x[0]**2 + 4*x[1]**2

def grad(x):
    return np.array([2*x[0], 8*x[1]])

x = np.array([1., 3.])
p = -grad(x)
a = line_search(objective, grad, x, p)[0]
print("MY ANSWER: ", backtrack(objective, grad, x, p))
print("NUMPY's ANSWER: ", a)

