# -*- coding: utf-8 -*-
import numpy as np
help(print)

'''Problem 1'''
A = np.array([[3, -1, 4],[1, 5, -9]])
B = np.array([[2, 6, -5, 3], [5, -8, 9, 7], [9, -3, -2, -3]])

Answer1 = A@B

print('\nProblem 1 Answer\n')
print(Answer1)

'''Problem 2'''
A_2 = np.array([[3, 1, 4],[1, 5, 9],[-5, 3, 1]])

Answer2 = -A_2@A_2@A_2 + 9*A_2@A_2 - 15 * A_2

print('\nProblem 2 Answer\n')
print(Answer2)

'''Problem 3'''
A_3 = np.triu(np.ones((7,7)))
b1 = np.tril(np.full_like(np.ones((7,7)), -1))
b2 = np.triu(np.full_like(np.ones((7,7)), 5)) - 5*np.eye(7)
B_3 = b1 + b2

Answer3 = A_3@B_3@A_3
Answer3 = Answer3.astype(np.int64)

print('\nProblem 3 Answer\n')
for line in Answer3:
    print(line)


'''Problem 4'''
A = np.array([[0,1,-2,3,4],[5,6,7,8,-9]])
A[1,1]

def return_0_for_negative(n):
    copy_matrix = np.copy(n)
    mask = copy_matrix < 0
    copy_matrix[mask] = 0
    return copy_matrix

print('\nProblem 4 Answer\n')
print(return_0_for_negative(A))

'''Problem 5'''
A_5 = np.arange(6).reshape((3,-1)).T
B_5 = np.tril(np.full_like(np.eye(3),3))
C_5 = np.eye(3) * -2
zero3x3 = np.full_like(np.ones((3,3)), 0)
zero3x2 = np.full_like(np.ones((3,2)), 0)
zero2x2 = np.full_like(np.ones((2,2)), 0)
zero2x3 = np.full_like(np.ones((2,3)), 0)

Answer5 = np.hstack(( np.vstack((zero3x3, A_5, B_5)), np.vstack((A_5.T, zero2x2, zero3x2)), np.vstack((np.eye(3), zero2x3, C_5)) ))

print('\nProblem 5 Answer\n')
for line in Answer5:
    print(line)
    
'''Problem 6'''
A_6 = np.array([[1,1,0],[0,1,0],[1,1,1]])

def prob6(n):
    n = n.astype(np.float64)
    row_sum = n.sum(axis=1)
    result = n/row_sum.reshape((row_sum.size,-1))
    return result

Answer6 = prob6(A_6)

print('\nProblem 6 Answer\n')
print(Answer6)
    
'''Problem 7'''
grid = np.load("grid.npy")

horizon = np.max(grid[:,:-3] * grid[:,1:-2] * grid[:, 2:-1] * grid[:,3:])
vertical = np.max(grid[:-3,:] * grid[1:-2,:] * grid[2:-1,:] * grid[3:,:])
right_diag = np.max(grid[:-3,:-3] * grid[1:-2,1:-2] * grid[2:-1, 2:-1] * grid[3:,3:])
left_diag = np.max(grid[3:,:-3] * grid[2:-1,1:-2] * grid[1:-2, 2:-1] * grid[:-3,3:])

Answer7 = max(horizon, vertical, right_diag, left_diag)

print('\nProblem 7 Answer\n')
print(Answer7)




















