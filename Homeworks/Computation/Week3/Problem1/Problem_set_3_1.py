import numpy as np
from scipy import linalg as la
import math

# Problem 1

print("\n\n --Problem1--\n\n")

# The function of Modified_GS
def Modified_GS(A):
    m, n = np.shape(A)
    Q = np.copy(A)
    R = np.zeros((n, n))
    
    for i in range(n):
        R[i, i] = la.norm(Q[:,i])
        Q[:, i] = Q[:, i] / R[i, i]
        
        j = i + 1
        for k in range( (n-1) - (i+1) + 1 ):
            R[i, j] = np.dot( Q[:, j].T, Q[:, i] )
            Q[:, j] = Q[:, j] - R[i, j]*Q[:, i]
            j += 1
    return Q, R

''' 
for my pratice

for j in range(1,5):
    print(j)
    
k = 1
for j in range( 4 - 1 + 1 ):
    print(k)
    k += 1
'''
            

# Generate A and return Q_mine and R_mine.
m = 6
n = 4
A = np.random.random((m,n))
Q_mine, R_mine = Modified_GS(A)

# Generate a random matrix and get its reduced QR decomposition via SciPy.
Q,R = la.qr(A, mode="economic") # Use mode="economic" for reduced QR.
print (A.shape, Q.shape, R.shape)

# Verify that R is upper triangular, Q is orthonormal, and QR = A.
print( np.allclose(np.triu(R), R) )
print( np.allclose(np.dot(Q.T, Q), np.identity(n)) )
print( np.allclose(np.dot(Q, R), A) )

# So, my value is correct here?
print( np.allclose(np.triu(R_mine), R_mine) )
print( np.allclose(np.dot(Q_mine.T, Q_mine), np.identity(n)) )
print( np.allclose(np.dot(Q_mine, R_mine), A) )


# Problem 2

# Determinant Function, one line.
def det(A_2, Q_A_2, R_A_2, n):
    detA_abs = abs( np.product([R_A_2[i, i] for i in range(n)]))
    return detA_abs

# New matrix declared.
m_2 = 2
n_2 = 2
A_2 = np.random.random((m_2,n_2))
Q_A_2, R_A_2 = Modified_GS(A_2)
detA_abs = det(A_2, Q_A_2, R_A_2, n_2)

# Compare my determinant with true determinant
print("\n\n --Problem2--\n")
print( np.allclose(abs(np.linalg.det(A_2)), detA_abs) )


'''
for my practice

x = [1, 2, 3, 4, 5]
y = [2*a for a in x] if a % 2 == 1]
print(y)

np.product([1,3,5])
'''

# Problem 3
print("\n\n --Problem3--\n")

# 1. New matrix declared. Compute Q and R
m_3 = 6
n_3 = 4
A_3 = np.random.random((m_3, n_3))
Q_A_3, R_A_3 = Modified_GS(A_3)
detA_abs = det(A_3, Q_A_3, R_A_3, n_3)
b = np.random.random((m_3,1))

# 2. y = Q.T*b
y = Q_A_3.T @ b

# 3. Back substitution for upper triangular matrix
def upperSol_back(A, b):
    n = np.size(b)
    x = np.zeros_like(b)

    x[-1] = b[-1] / A[-1, -1]
    
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - np.sum(A[i, i+1:] * x[i+1:].T) ) / A[i, i] 

    return x

# Answer
print("x =", upperSol_back(R_A_3, y))
print("x =", la.solve_triangular(R_A_3, y))


# Problem 4

# I will use matrix A from problem 1.
print("\n\n --Problem4--\n")

# Householder 

def sign(a):
    sign = lambda x: 1 if x >= 0 else -1
    sign_a = sign(a)
    return sign_a

def Householder(A):
    m, n = np.shape(A)
    R = np.copy(A)
    Q = np.zeros((m, m))
    
    for k in range(n):
        u = np.copy(R[k:, k])
        u[0] = u[0] + sign(u[0]) * np.linalg.norm(u)
        u = u / np.linalg.norm(u)
        R[k:, k:] = R[k:,k:] - 2*np.outer(u, u.T @ R[k:,k:])
        Q[k:,:] = Q[k:,:] - 2*np.outer(u, u.T @ Q[k:,:])
    return Q.T, R


Q_4, R_4 = la.qr(A)

print (A.shape, Q_4.shape, R_4.shape)

# Verify that R is upper triangular, Q is orthonormal, and QR = A.
print( np.allclose(np.triu(R_4), R_4) )
print( np.allclose(np.dot(Q_4.T, Q_4), np.identity(m)) ) 
print( np.allclose(np.dot(Q_4, R_4), A) )
 
# Problem 5
print("\n\n --Problem5--\n")

def Hessen(A):
    m, n = np.shape(A)
    H = np.copy(A)
    Q = np.identity(m)
    
    for k in range(n-2):
        u = np.copy(H[k+1:,k])
        u[0] = u[0] + sign(u[0]) * np.linalg.norm(u)
        u = u / np.linalg.norm(u)
        H[k+1:,k:] = H[k+1:,k:] - 2 * np.outer(u, u.T@H[k+1:,k:])
        H[:,k+1:] = H[:, k+1:] - 2 * np.outer(H[:,k+1:]@u, u.T)
        Q[k+1:,:] = Q[k+1:,:] - 2 * np.outer(u, u.T@Q[k+1:,:])
    return H, Q.T
    
# declare new matrix
A_5 = np.random.random((8,8))

# Calculate H and Q
H_5_mine, Q_5_mine = Hessen(A_5)
H_5, Q_5 = la.hessenberg(A_5, calc_q=True)

# Test
print(np.allclose(H_5, H_5_mine))
print(np.allclose(Q_5, Q_5_mine))

print(np.allclose(np.triu(H_5_mine, -1), H_5_mine))
print(np.allclose(np.dot(np.dot(Q_5_mine, H_5_mine), Q_5_mine.T), A_5))














