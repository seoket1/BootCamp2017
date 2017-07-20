from scipy import linalg as la
import numpy as np
from matplotlib import pyplot as plt
import cmath
import math
from scipy.sparse import dok_matrix
import sympy as sy
from matplotlib import pyplot as plt
from numba import jit
from sympy.abc import x
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy import Function
from numba import jit
from autograd import grad
import autograd.numpy as anp
from autograd import jacobian
from autograd import elementwise_grad
import time

# Problem 1
print("\n\n Problem 1 \n")
def f_prime(Functions, X):
    
    original_fn = []
    prime_fn = []
    for i in range(len(X)):
        x = sy.symbols('x')
        fx = lambda x: Functions
        lam_f_x = lambdify(x, fx(x))
        lam_f_prime_x = lambdify(x, sy.diff(fx(x)))
        original_fn.append(lam_f_x(X[i]))
        prime_fn.append(lam_f_prime_x(X[i]))
    return original_fn, np.array(prime_fn), lam_f_x, lam_f_prime_x, X

X = np.linspace(-np.pi, np.pi, 200)
x = sy.symbols('x')
Functions = (sy.sin(x) + 1)**sy.sin(sy.cos(x))
original_fn, prime_fn, lam_f_x, lam_f_prime_x, X = f_prime(Functions, X)

plt.plot(X, original_fn)
plt.plot(X, prime_fn, color = "red")
ax = plt.gca()
ax.spines["bottom"].set_position("zero")
plt.show()

# Problem 2
print("\n\n Problem 2 \n")
def f_prime_different(Functions, X, h):
    original_fn = []
    prime_fn = []
    foward1 = []
    foward2 = []
    backward1 = []
    backward2 = []
    centered1 = []
    centered2 = []
    
    for i in range(len(X)):
        x = sy.symbols('x')
        fx = lambda x: Functions
        lam_f_x = lambdify(x, fx(x))
        
        foward1.append( ( lam_f_x(X[i] + h) - lam_f_x(X[i]) ) / h )
        foward2.append( ( -3*lam_f_x(X[i]) + 4*lam_f_x(X[i] + h) - lam_f_x(X[i] + 2*h) ) / (2*h) )
        backward1.append( ( lam_f_x(X[i]) - lam_f_x(X[i] - h) ) / h )
        backward2.append( ( 3*lam_f_x(X[i]) - 4*lam_f_x(X[i] - h) + lam_f_x(X[i] - 2*h) ) / (2*h) )
        centered1.append( ( lam_f_x(X[i] + h) - lam_f_x(X[i] - h) ) / (2*h) )
        centered2.append( ( lam_f_x(X[i] - 2*h) - 8*lam_f_x(X[i] - h) + 8*lam_f_x(X[i] + h) - lam_f_x(X[i] + 2*h) ) / (12*h) )
        
        original_fn.append(lam_f_x(X[i]))
        
    return original_fn, np.array(foward1), np.array(foward2), np.array(backward1), np.array(backward2), np.array(centered1), np.array(centered2)


Functions = (sy.sin(x) + 1)**sy.sin(sy.cos(x))
X = np.linspace(-np.pi, np.pi, 200)
original_fn, foward1, foward2, backward1, backward2, centered1, centered2 = f_prime_different(Functions, X, h = 1e-10)



plt.plot(X, original_fn, label = "orginal function")
plt.plot(X, prime_fn, color = "red", label = "prime fn from problem1")
plt.plot(X, foward1, label = "foward1") # pass
plt.plot(X, foward2, label = "foward2")
plt.plot(X, backward1, label = "backward1") # pass
plt.plot(X, backward2, label = "backward2")
plt.plot(X, centered1, label = "centered1")
plt.plot(X, centered2, label = "centered2")
plt.legend()
ax = plt.gca()
ax.spines["bottom"].set_position("zero")
plt.show()

# Problem 3
print("\n\n Problem 3\n")
@jit
def problem3(X, h):
    Functions = (sy.sin(x) + 1)**sy.sin(sy.cos(x))
    original_fn, prime_fn, lam_f_x, lam_f_prime_x, X = f_prime(Functions, X)

    for1 = []
    for2 = []
    back1 = []
    back2 = []
    cent1 = []
    cent2 = []
    
    for i in range(len(h)):
        original_fn, foward1, foward2, backward1, backward2, centered1, centered2 = f_prime_different(Functions, X, h[i])
        for1.append(la.norm(prime_fn - foward1))
        for2.append(la.norm(prime_fn - foward2))
        back1.append(la.norm(prime_fn - backward1))
        back2.append(la.norm(prime_fn - backward2))
        cent1.append(la.norm(prime_fn - centered1))
        cent2.append(la.norm(prime_fn - centered2))
    return for1, for2, back1, back2, cent1, cent2

X0 = [1]
h = np.logspace(-8, 0, 9)

for1, for2, back1, back2, cent1, cent2 = problem3(X0, h)

plt.loglog(h, for1, label = "for1")
plt.loglog(h,for2, label = "for2")
plt.loglog(h,back1, label = "back1")
plt.loglog(h,back2, label = "back2")
plt.loglog(h,cent1, label = "cent1")
plt.loglog(h,cent2, label = "cent2")
plt.legend()
plt.show()

# Problem 4
print("\n\n Problem 4 \n")
def radar():
    data = np.load("C:/Users/suket/Desktop/Homeworks/Computation/Week4/Problem3/plane.npy")
    alpha = np.deg2rad(data[:, 1])
    beta = np.deg2rad(data[:, 2])

    X_Functions = lambda x, y : 500 * (np.tan(y)) / ( np.tan(y) - np.tan(x) )
    Y_Functions = lambda x, y : 500 * (np.tan(y) * np.tan(x)) / ( np.tan(y) - np.tan(x) )
    XY_Functions = lambda x_prime, y_prime : (x_prime ** 2 + y_prime ** 2) ** 0.5
    
    speed = []
    for t in range(7, 15, 1):
        if t == 7:
            x_prime = ((X_Functions(alpha[t-6], beta[t-6]) - X_Functions(alpha[t-7], beta[t-7])) / 1)
            y_prime = ((Y_Functions(alpha[t-6], beta[t-6]) - Y_Functions(alpha[t-7], beta[t-7])) / 1)
            speed.append( XY_Functions(x_prime, y_prime) )
            
        elif t > 7 and t < 14:
            x_prime = ((X_Functions(alpha[t-6], beta[t-6]) - X_Functions(alpha[t-8], beta[t-8])) / 2)
            y_prime = ((Y_Functions(alpha[t-6], beta[t-6]) - Y_Functions(alpha[t-8], beta[t-8])) / 2)
            XY_Functions(x_prime, y_prime)
            speed.append( XY_Functions(x_prime, y_prime) )
        else:
            x_prime = ((X_Functions(alpha[t-7], beta[t-7]) - X_Functions(alpha[t-8], beta[t-8])) / 1)
            y_prime = ((Y_Functions(alpha[t-7], beta[t-7]) - Y_Functions(alpha[t-8], beta[t-8])) / 1)
            XY_Functions(x_prime, y_prime)
            speed.append( XY_Functions(x_prime, y_prime) )
    
    return speed

print(radar())

# Problem 5
print("\n\n Problem 5 \n")
def Jacobian(x_5, fx_5, h = 1e-10):
    Jacob = np.zeros((np.size(x_5), np.size(x_5)))
    I = np.identity(np.size(x_5))
    for i in range(np.size(x_5)):
        for j in range(np.size(fx_5)):
            Jacob[j, i] = (fx_5[j](x_5 + h*I[:, i]) - fx_5[j](x_5 - h*I[:, i])) / (2*h) 
    return Jacob

fx_5_1 = lambda x: x[0]**2
fx_5_2 = lambda x: x[0]**3 - x[1]

x_5 = np.array([5, 10])
fx_5 = np.array([fx_5_1, fx_5_2])

print(Jacobian(x_5, fx_5, 1e-10))

# Problem 6
print("\n\n Problem 6 \n")
def sympy_method(X, fx):
    x = sy.symbols('x')
    lam_f_x = lambdify(x, fx(x))
    lam_f_prime_x = lambdify(x, sy.diff(fx(x)))
    return lam_f_prime_x(X)

def second_order_method(x, f, h = 1e-10):
    secone_order = (f(x + h) - f(x - h)) / (2*h)
    return secone_order

def autograd(x, f):
    grad_g = grad(f)
    return grad_g(x)

f_6 = lambda x: sy.log(sy.sin(sy.sqrt(x)))
x_6 = sy.pi/4
start_time = time.clock()
print("Answer =", sympy_method(x_6, f_6))
print(time.clock() - start_time)

f_6 = lambda x: np.log(np.sin(np.sqrt(x)))
x_6 = np.pi/4
start_time = time.clock()
print("Answer =", second_order_method(x_6, f_6))
print(time.clock() - start_time)

start_time = time.clock()
f_6 = lambda x: anp.log(anp.sin(anp.sqrt(x)))
x_6 = anp.pi/4
print("Answer =", autograd(x_6, f_6))
print(time.clock() - start_time)

# Problem 7
print("\n\n Problem 7 \n")
# Define the Taylor series.
# Note that this function does not account for array broadcasting.
def taylor_exp_for_sin(x, N = 10000, tol=.0001):
    result = 0
    cur_term = x
    i = 0
    while anp.abs(cur_term) >= tol:
        # Autograd's version of NumPy doesn't have the math attribute so use NumPy.
        cur_term = ( (-1)**i/np.math.factorial(2*i + 1) ) * x**(2*i + 1)
        result += cur_term
        i += 1
    return result

def calculate_derivative(x0, N):
    # Compute the gradient.
    if N == 1:
        d_taylor_exp = grad(taylor_exp_for_sin)
    elif N ==2:
        d_taylor_exp = grad(grad(taylor_exp_for_sin))
    else:
        return False
    # Note that differentiation in autograd only works with float values.
    deri_f = []
    for i in range(len(x0)):
        deri_f.append(d_taylor_exp(x0[i], N))
    return deri_f

x0 = np.linspace(-np.pi, np.pi , 500)

deri_f = calculate_derivative(x0, 1)
plt.plot(x0, np.sin(x0))
plt.plot(x0, deri_f)
ax = plt.gca()
ax.spines["bottom"].set_position("zero")
plt.show()

deri_f = calculate_derivative(x0, 2)
plt.plot(x0, np.sin(x0))
plt.plot(x0, deri_f)
ax = plt.gca()
ax.spines["bottom"].set_position("zero")
plt.show()

# Problem 8
print("\n\n Problem 8 \n")

def sympy_jacob(x0, y0, X, Y):
    x = sy.symbols('x')
    y = sy.symbols('y')
    lam_f_x = lambdify((x, y), Y.jacobian(X))
    return lam_f_x(x0, y0)

def auto_jacob(x, f):
    jacob_f = jacobian(f)
    return jacob_f(x_8)

#from sympy.abc import x, y
x = sy.symbols('x')
y = sy.symbols('y')
Y = sy.Matrix([sy.exp(x) * sy.sin(y) + y**3, 3*y - sy.cos(x)])
X = sy.Matrix([x, y])
x_8_1 = 1
x_8_2 = 1
start_time = time.clock()
print("Answer =", sympy_jacob(x_8_1, x_8_2, X, Y))
print(time.clock() - start_time)

f_8_1 = lambda x : np.exp(x[0]) * np.sin(x[1]) + x[1]**3
f_8_2 = lambda x : 3*x[1] - np.cos(x[0])
x_8 = np.array([1, 1])
f_8 = np.array([f_8_1, f_8_2])
start_time = time.clock()
print("\nAnswer =", Jacobian(x_8, f_8))
print(time.clock() - start_time)

f_8 = lambda x : anp.array([anp.exp(x[0]) * anp.sin(x[1]) + x[1]**3, 3*x[1] - anp.cos(x[0])])
x_8 = anp.array([1.0, 1.0])
print("\nAnswer =", auto_jacob(x_8, f_8))
print(time.clock() - start_time)















