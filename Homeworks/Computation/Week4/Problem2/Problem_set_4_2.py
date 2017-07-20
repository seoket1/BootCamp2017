from scipy import linalg as la
import numpy as np
from matplotlib import pyplot as plt
import cmath
import math
from scipy.sparse import dok_matrix
import sympy as sy
from matplotlib import pyplot as plt
from numba import jit


# Problem 1
def k_A(A):
    U, s, Vh = la.svd(A)
    if max(s)/min(s) == np.inf:
        return np.inf
    else:
        k_A = max(s)/min(s) 
        return k_A

m = 6
n = 4

A1 = np.random.random((m,n))
print(k_A(A1))
A2 = np.array([[1, 0], [0, 0]])
print(k_A(A2))

print(np.allclose(np.linalg.cond(A1), k_A(A1)))

# Problem 2
def experiment():
    absolute = np.zeros(100)
    relative = np.zeros(100)
    new_roots_set = np.zeros((20, 100), dtype=np.complex)
    
    # The roots of w are 1, 2, ..., 20.
    w_roots = np.arange(1, 21)
    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())
    r = np.zeros(21)
    
    for i in range(100):
        for j in range(21):
            r[j] = np.random.normal(1, 10**(-10))
        new_coeffs = np.copy(w_coeffs * r)
        # Use NumPy to compute the roots of the perturbed polynomial.
        new_roots = np.roots(np.poly1d(new_coeffs))
        # Sort the roots to ensure that they are in the same order.
        w_roots = np.sort(w_roots)
        new_roots = np.sort(new_roots)
        # Estimate the absolute condition number in the infinity norm.
        h = new_coeffs - w_coeffs
        absolute[i] = la.norm(new_roots - w_roots, np.inf) / la.norm(h, np.inf)
        # Estimate the relative condition number in the infinity norm.
        relative[i] = absolute[i] * la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf) # This is huge!!
        
        new_roots_set[:, i] = new_roots
        
    absol = np.sum(absolute)/100
    rela = np.sum(relative)/100

    return w_roots,  new_roots_set, absol, rela

w_roots, new_roots_set, absolute, relative = experiment()

plt.scatter(new_roots_set.real, new_roots_set.imag, marker = ',', color = 'black', s = 1)
plt.scatter(w_roots.real, w_roots.imag)
plt.show()

# Problem 3

def condition_for_matrix(A):
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags

    la.eigvals(A+H)

    absolute = la.norm( la.eigvals(A) - la.eigvals(A+H) ) / la.norm(H)
    relative = absolute * la.norm(A)/la.norm(la.eigvals(A))
    return absolute, relative

m = 4
n = 4
A3 = np.random.random((m,n))
condition_for_matrix(A3)

# Problem 4
@jit
def gray_r(xmin, xmax, ymin, ymax, res):
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)

    A4 = np.zeros((2,2))
    A4[0,0] = 1
    A4[1,1] = 1
    
    for_rela = np.zeros((res,res))
    for i in range(len(x)):
        for j in range(len(y)):
            A4[0,1] = x[i]
            A4[1,0] = x[j]
            absol, rela = condition_for_matrix(A4)
            for_rela[i,j] = rela
    return x, y, for_rela

x, y, for_rela = gray_r(-100,100,-100,100,200)

plt.pcolormesh(x, y, for_rela, cmap="gray_r")
plt.colorbar()
plt.show()
        
# Problem 5
n=14
xk, yk = np.load("C:/Users/suket/Desktop/Homeworks/Computation/Week4/Problem2/stability_data.npy").T
A5 = np.vander(xk, n+1)

def least_squares(A, b): 
    # Generate a random matrix and get its reduced QR decomposition via SciPy.
    Q, R = la.qr(A, mode="economic") # Use mode="economic" for reduced QR.
    
    # y = RX = Q.T*b
    y = Q.T @ b 
    
    # Answer
    x = la.solve_triangular(R, y)
    return x

def comparison(A):
    coef_way_1 = la.inv(A.T@A)@A.T@yk
    coef_way_2 = least_squares(A, yk)
    
    
    plt.scatter(xk,yk, label = 'original data', s = 4)
    plt.plot(xk, np.polyval(coef_way_1, xk), color = "red", label = 'normal')
    plt.plot(xk, np.polyval(coef_way_2, xk), color = "black", label = "QR")
    plt.legend()
    plt.ylim(0,25)
    plt.show()
    
    forward_error_1 = la.norm(np.polyval(coef_way_1, xk) - yk)
    forward_error_2 = la.norm(np.polyval(coef_way_2, xk) - yk)
    return forward_error_1, forward_error_2

forward_error_1, forward_error_2 = comparison(A5)

# Problem 6
def In(n):
    relative_forward_error = []
    for i in range(len(n)):
        x, y = sy.symbols('x,y')
        In_1 = float(sy.integrate((x**int(n[i])) * sy.exp(x - 1), (x,0,1)))
        In_2 = float( ((-1)**n[i]) *sy.subfactorial(int(n[i])) + ((-1)**(n[i]+1)) * ( sy.factorial(int(n[i]))/sy.exp(1) ) )
        relative_forward_error.append(abs(In_2 - In_1) / abs(In_1))
    return relative_forward_error

n = np.linspace(0, 50, 11)
forward_error = In(n)

plt.title("error measure")
plt.plot(n, forward_error)
plt.ylim(0,1000)
plt.show()

print("This method is not stable as you can see in the graph. From 20, its error size becomes so large.")







