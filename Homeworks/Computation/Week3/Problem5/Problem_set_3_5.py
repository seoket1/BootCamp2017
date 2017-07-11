import matplotlib.pyplot as plt
import numpy as np
from numba import jit

p, q = 0.1, 0.2  # Prob of leaving low and high state respectively

def compute_series(n):
    x = np.empty(n, dtype=int)
    x[0] = 1  # Start in state 1
    U = np.random.uniform(0, 1, size=n)
    for t in range(1, n):
        current_x = x[t-1]
        if current_x == 0:
            x[t] = U[t] < p
        else:
            x[t] = U[t] > q
    return x

n = 100000
x = compute_series(n)
print(np.mean(x == 0))  # Fraction of time x is in state 0

%timeit compute_series(n)

compute_series_numba = jit(compute_series)

x = compute_series_numba(n)
print(np.mean(x == 0))

%timeit compute_series_numba(n)

%load_ext Cython
%%cython
import numpy as np
from numpy cimport int_t, float_t

def compute_series_cy(int n):
    # == Create NumPy arrays first == #
    x_np = np.empty(n, dtype=int)
    U_np = np.random.uniform(0, 1, size=n)
    # == Now create memoryviews of the arrays == #
    cdef int_t [:] x = x_np
    cdef float_t [:] U = U_np
    # == Other variable declarations == #
    cdef float p = 0.1
    cdef float q = 0.2
    cdef int t
    # == Main loop == #
    x[0] = 1
    for t in range(1, n):
        current_x = x[t-1]
        if current_x == 0:
            x[t] = U[t] < p
        else:
            x[t] = U[t] > q
    return np.asarray(x)

compute_series_cy(10)

x = compute_series_cy(n)
print(np.mean(x == 0))

%timeit compute_series_cy(n)
