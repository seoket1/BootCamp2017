from scipy import linalg as la
from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt
import cmath
import math
from scipy.sparse import dok_matrix
import sympy as sy
import scipy.integrate
from matplotlib import pyplot as plt
from numba import jit
from autograd import grad
import autograd.numpy as anp
from autograd import jacobian
from scipy.stats import norm
import random

# Problem 1
print("\n\nProblem 1\n")
def integral(a, b, f, h, method):
    if method == "midpoint":
        h = float(b-a)/h
        summation = 0.0
        x = a + h/2            
        while (x < b):
            summation += h * f(x)
            x += h
    elif method == "trapezoid":
        b_a = float(b-a)/h
        summation = (b_a/2) * (f(a) + f(b))
        for i in range(1, h):
            summation += b_a * f(a + i*b_a)
    elif method == "simpsons":
        b_a = float(b - a) / h
        summation = f(a) + f(b)
        for i in range(1, h, 2):
            summation += 4 * f(a + i * b_a)
        for i in range(2, h-1, 2):
            summation += 2 * f(a + i * b_a)
        summation = summation * b_a / 3
    return summation

f = lambda x: 0.1*x**4 - 1.5*x**3 + 0.53*x**2 + 2*x +1

mid = integral(-10, 10, f, 10000, "midpoint")
trape = integral(-10, 10, f, 10000, "trapezoid")
sim = integral(-10, 10, f, 10000, "simpsons")

print("mid =", mid)
print("trape =", trape)
print("sim =", sim)

# Problem 2
print("\n\nProblem 2\n")
def discrete_normal(mu, sigma, N, k):
    discrete = np.linspace(mu - k*sigma, mu + k*sigma, N)
    bins = discrete[1] - discrete[0]
    w = norm.cdf(discrete + 0.5*bins, loc = mu, scale = sigma) - norm.cdf(discrete - 0.5*bins, loc = mu, scale = sigma)
    return w, discrete

print(discrete_normal(0, 1, 11, 3))

# Problem 3
print("\n\nProblem 3\n")
def discrete_lognormal(mu, sigma, N, k):
    discrete = np.linspace(mu - k*sigma, mu + k*sigma, N)
    bins = discrete[1] - discrete[0]
    w = norm.cdf(discrete + 0.5*bins, loc = mu, scale = sigma) - norm.cdf(discrete - 0.5*bins, loc = mu, scale = sigma)
    exp_discrete = np.exp(discrete)
    return w, exp_discrete

print(discrete_lognormal(0, 1, 11, 3))

# Problem 4
print("\n\nProblem 4\n")
w, discrete = discrete_lognormal(10.5, 0.8, 11, 3)

expected_value_lognormal = w @ discrete
expected_value = np.exp(10.5 + (0.8**2)/2)

print("difference = ", expected_value_lognormal - expected_value )

# Problem 5
print("\n\nProblem 5\n")
from scipy.special.orthogonal import p_roots
def gauss(f,n,a,b):
    [x,w] = p_roots(n+1)
    answer = 0.5*(b-a) * sum( w*f(0.5*(b-a)*x+0.5*(b+a)))
    return answer

f = lambda x : 0.1*x**4 - 1.5*x**3 + 0.53*x**2 + 2*x + 1
print(gauss(f, 3, -10, 10))

# Problem 6
print("\n\nProblem 6\n")
print(scipy.integrate.quad(f, -10, 10))

# Problem 7
print("\n\nProblem 7\n")

@jit
def pi(N):
    count = 0
    for i in range(N):
      x = 2*random.random() - 1
      y = 2*random.random() - 1
      if x**2 + y**2 <= 1.0:
         count += 1
    pi = (float(count) / N) * 4     
    return pi

print(pi(int(1e+5)))

# from 1e+9, it goes to the true value.

# Problem 8
print("\n\nProblem 8\n")
def get_prime_numbers(n):
    if n <= 2:
        raise StopIteration
    yield 2
    for i in range(3, n, 2):
        for x in range(3, int(i**0.5) + 2, 2):
            if not i % x:
                break
        else:
            yield i

def numbering_prime(N):
    prime = []
    for i in get_prime_numbers(int(1e+10)):
        prime.append(i)
        if len(prime) == N:
            return prime
        
def Weyl(n, s):
    seq = np.zeros((s, n))
    p = numbering_prime(s)
    for i in range(1, n + 1):
        for j in range(0, s):
            seq[j, i - 1] = (i * p[j]**0.5) - math.floor(i * p[j]**0.5)
    return seq

def Haber(n, s):
    seq = np.zeros((s, n))
    p = numbering_prime(s)
    for i in range(1, n + 1):
        for j in range(0, s):
            seq[j, i - 1] = (i*(1+i)/2*p[j]**0.5) - math.floor(i*(1+i)/2*p[j]**0.5) 
    return seq

def Niederreiter(n, s):
    seq = np.zeros((s, n))
    for i in range(1, n + 1):
        for j in range(1, s + 1):
            seq[j - 1, i - 1] = ((i) * 2. ** ((j) / (j + 1.)) -
                                 math.floor((i) * 2. **
                                 ((j) / (j + 1.))))
    return seq

def Baker(n, s):
    seq = np.zeros((s,n))
    for i in range(1, n+1):
        for j in range(1, s+1):
            seq[j-1, i-1] = ((i+1)*np.exp(j + 1)) - math.floor((i + 1) *
                    np.exp(j + 1))
    return seq

# Problem 9
print("\n\nProblem 9\n")
@jit
def pi_different_way(N, method):
    count = 0
    if method == "Weyl":
        x = Weyl(int(N), 1)
        y = Weyl(int(N), 2)
    elif method == "Haber":
        x = Haber(int(N), 1)
        y = Haber(int(N), 2)
    elif method == "Niederreiter":
        x = Niederreiter(int(N), 1)
        y = Niederreiter(int(N), 2)    
    elif method == "Baker":
        x = Baker(int(N), 1)
        y = Baker(int(N), 2)    
        
    for i in range(int(N)):
        if x[0, i]**2 + y[1, i]**2 <=1:
            count += 1
    pi = (float(count) / N) * 4     
    return pi
    
N = int(1e+5)
print(pi_different_way(1e+5, "Weyl"))
print(pi_different_way(1e+5, "Haber"))
print(pi_different_way(1e+5, "Niederreiter"))
print(pi_different_way(1e+5, "Baker"))


# It is fater than Problem 7.Backer method was the best when I did experiments.





