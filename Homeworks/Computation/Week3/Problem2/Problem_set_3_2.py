import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from scipy.stats import linregress
import math
from scipy import linalg as la
import cmath

# Problem 1
print("\n\n --Problem 1--\n")

def least_squares(A, b): 
    # Generate a random matrix and get its reduced QR decomposition via SciPy.
    Q, R = la.qr(A, mode="economic") # Use mode="economic" for reduced QR.
    
    # y = RX = Q.T*b
    y = Q.T @ b 
    
    # Answer
    x = la.solve_triangular(R, y)
    return x

# Generate A and return Q_mine and R_mine.
m = 6
n = 4

A = np.random.random((m,n))
b = np.random.random((m,1))
print("x =", least_squares(A, b))


# Problem 2
print("\n\n --Problem 2--\n")
housing = np.load('C:/Users/suket/Desktop/Homeworks/Computation/Week3/Problem2/housing.npy')

# 1. Constrcut A and b
year = housing[:,0]
price_T = housing[:,1]
price = np.vstack(price_T)

m_2 = np.size(year)
constant_T = np.ones(m_2)
constant = np.vstack(constant_T)

A2 = np.column_stack((year, constant))

# 2. least squares
x_2 = least_squares(A2, price)
a2, b2, rvalue2, pvalue2, stderr2 = linregress(year, price_T)

# Plotting
plt.scatter(year, price, label = "Scatter")
plt.plot(year, A2@x_2, label = "Mine")
plt.plot(year, a2*year + b2, label = "Built-in Function")
plt.show()

'''
For Practice!

# Generate some random data close to the line y = .5x - 3.
x = np.linspace(0, 10, 20)
y = .5*x - 3 + np.random.randn(20)
# Use linregress() to calculate m and b, as well as the correlation
# coefficient, p-value, and standard error. See the documentation for
# details on each of these extra return values.
a, b, rvalue, pvalue, stderr = linregress(x, y)
plt.plot(x, y, 'k*', label="Data Points")
plt.plot(x, a*x + b, 'b-', lw=2, label="Least Squares Fit")
plt.legend(loc="upper left")
plt.show()
'''

# Problem 3
print("\n\n --Problem 3--\n")
# Contruct A matrix
A3_degree_3 = np.vander(year, 4)
A3_degree_6 = np.vander(year, 7)
A3_degree_9 = np.vander(year, 10)
A3_degree_12 = np.vander(year, 13)

# Regression
x_degree_3 = la.lstsq(A3_degree_3, price)[0]
x_degree_6 = la.lstsq(A3_degree_6, price)[0]
x_degree_9 = la.lstsq(A3_degree_9, price)[0]
x_degree_12 = la.lstsq(A3_degree_12, price)[0]

# Plotting

plt.subplot(221)
plt.scatter(year, price, label = "Scatter")
plt.plot(year, A3_degree_3@x_degree_3, label = "degree_3")
plt.subplot(222)
plt.scatter(year, price, label = "Scatter")
plt.plot(year, A3_degree_6@x_degree_6, label = "degree_6")
plt.subplot(223)
plt.scatter(year, price, label = "Scatter")
plt.plot(year, A3_degree_9@x_degree_9, label = "degree_9") 
plt.subplot(224)       
plt.scatter(year, price, label = "Scatter")
plt.plot(year, A3_degree_12@x_degree_12, label = "degree_12")
plt.show()

# Compare np.polyfit()
x_fit_3 = np.polyfit(year, price, 3)
x_fit_6 = np.polyfit(year, price, 6)
x_fit_9= np.polyfit(year, price, 9)
x_fit_12 = np.polyfit(year, price, 12)

print(np.allclose(x_fit_3, x_degree_3))
print(np.allclose(x_fit_6, x_degree_6))
print(np.allclose(x_fit_9, x_degree_9))
print(np.allclose(x_fit_12, x_degree_12), ": when we assume higher degree,\
 we can face a trouble like this.") 
# when we have higher degree, we can face trouble like this.


# Problem 4
print("\n\n --Problem 4--\n")
def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A))/(2*A)
    
    plt.plot(r*cos_t, r*sin_t, lw=2)
    plt.gca().set_aspect("equal", "datalim")

ellipse = np.load('C:/Users/suket/Desktop/Homeworks/Computation/Week3/Problem2/ellipse.npy')

plt.scatter(ellipse[:,0], ellipse[:,1])
x_4_T = ellipse[:,0]
y_4_T = ellipse[:,1]
x_4 = np.vstack(x_4_T)
y_4 = np.vstack(y_4_T)

m_4 = np.size(x_4)
constant_T = np.ones(m_4)

A4 = np.column_stack((x_4*x_4, x_4, x_4*y_4, y_4, y_4*y_4))
slope = least_squares(A4, constant_T)

plot_ellipse(slope[0], slope[1], slope[2], slope[3], slope[4])
plt.show()


# Problem 5
print("\n\n --Problem 5--\n")
def Power_method(A, N, tol):
    m, n = np.shape(A)
    x_5 = np.random.random((m,1))
    x_5 = x_5 / np.linalg.norm(x_5)

    for k in range(N):
        x_5_1 = A @ x_5 
        x_5_1 = x_5_1 / np.linalg.norm(x_5_1)
        
        if np.linalg.norm(x_5_1 - x_5) < tol:
            break
        
        x_5 = x_5_1
        
    return x_5_1.T @ A @ x_5_1, x_5_1

A5 = np.random.random((10,10))
N = 10^5
tol = 1e-8
eigs, vecs = Power_method(A5, N, tol)

loc = np.argmax(eigs)
lamb, x = eigs[loc], vecs[:,loc]
print(np.allclose(A5.dot(x), lamb*x))


# Problem 6
print("\n\n --Problem 6--\n")

def QR_eig(A, N, tol):
    m, n = np.shape(A)
    S = la.hessenberg(A)
    
    for k in range(N):
        Q, R = la.qr(S)
        S = R@Q
    
    eigs = []
    i = 0
    
    while i < n:
        if  i == n-1 or abs(S[i+1, i]) < tol:
            eigs.append(S[i, i])
        else:
            a, b, c, d = S[i, i], S[i, i+1], S[i+1, i], S[i+1, i+1]
            lambda_plus = ( a+d + cmath.sqrt((a+d)**2 - 4*(a*d-b*c)) ) / 2
            lambda_minus = ( a+d - cmath.sqrt((a+d)**2 - 4*(a*d-b*c)) ) / 2 
            eigs.append(lambda_plus)
            eigs.append(lambda_minus)
            i = i + 1
        i = i + 1
    
    return eigs

A6 = np.random.random((5,5))
A6_sym = A6 + A6.T
N = 10^100
tol = 1e-10

eig_mine = QR_eig(A6_sym, N, tol)
eig_builtin = la.eig(A6_sym)[0]

print(eig_mine, "\n")
print(eig_builtin)






