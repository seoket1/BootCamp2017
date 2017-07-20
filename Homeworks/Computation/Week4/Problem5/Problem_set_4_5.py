from scipy import linalg as la
from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt
import cmath
import math
from scipy.sparse import dok_matrix
import sympy as sy
from matplotlib import pyplot as plt
from numba import jit
from autograd import grad
import autograd.numpy as anp
from autograd import jacobian
#from sympy.abc import x
#from sympy.utilities.lambdify import lambdify, implemented_function
#from sympy import Function

class SimplexSolver(object):
    
    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b
        self.m, self.n = A.shape
        
        origin = np.zeros(A.shape[1])
        is_not_feasible = (A @ origin > b).all()
        
        if is_not_feasible:
            raise ValueError('Origin is not in range of A.')
        else:
            # print('Origin is in range of A.')
            
            # introduce m slack variables when A is an mxn matrix 
            # one for each constraint equation
            
            #self.slack
            
            # keep track of the n variables
            # first the basic (i.e. non-zero) variables
            
            self.tracklist = list(range(self.n, self.m+self.n))
            self.tracklist += range(self.n)

        
    def make_tableau(self):
        A_bar = np.concatenate([A, np.identity(self.m)], axis=1)
        c_bar = np.concatenate([c, np.zeros(self.m)])

        tab1 = np.concatenate([np.array([0]), -1*c_bar, np.array([1])])
        tab2 = np.concatenate([b.reshape(self.m ,1), A_bar,
                               np.zeros((self.m, 1))], axis=1)

        self.tableau = np.concatenate([tab1.reshape(1, len(tab1)),
                                       tab2],
                                      axis=0)
    def blands_rule(self):
        # to determine the pivot column
        # find the first negativ element in the top row
        neg_values = np.where(self.tableau[0] < 0)
        entry = neg_values[0][0] 
        self.entry = entry        # equal to pivot column in tableau
        
        # find element in this column on which to pivot
        if (self.tableau[:,entry + 1] < 0).all():
            raise ValueError('Problem is unbounded. There exists no solution.')
        
        else:
            T_ks = list(np.where(self.tableau[:, entry ] > 0)[0])
            #row_last_neg_val = np.where(T_ks == True)[0][0] - 1
            #print(row_last_neg_val)
            self.T_ks = T_ks
            
            T_ratios = (self.tableau[T_ks, entry-1] / 
                        self.tableau[T_ks, entry])
            
            
            self.T_ratios = T_ratios
            self.leave = solv.T_ks[np.argmin(solv.T_ratios)] + 1
    
    def swap(self):
        self.tracklist[self.leave], self.tracklist[self.entry] = self.tracklist[self.entry], self.tracklist[self.leave]
            
    def pivot(self):
        pivot_value = self.tableau[self.leave-1, self.entry]
        # leave-1 corresponds to the pivot row in the tableau
        self.tableau[self.leave - 1,:] = (self.tableau[self.leave - 1,:] /
                        self.tableau[self.leave-1, self.entry])
        
        rows_to_zero_out = [x for x in range(len(self.tableau)) if x != self.leave - 1]
        
        factors = self.tableau[:,self.entry][rows_to_zero_out]
        
        self.tableau[[rows_to_zero_out]] -= np.outer(factors,
                                                     self.tableau[self.leave-1])
        # print(self.tableau)
        
    def one_step(self):
            
        self.blands_rule()
        self.swap()
        self.pivot()
        
    def solve(self):
        self.make_tableau()
        first_row_is_not_pos = (self.tableau[0] < 0).any()
        while first_row_is_not_pos:
            self.one_step()
            first_row_is_not_pos = (self.tableau[0] < 0).any()
        
        objective_value = self.tableau[0,0]
        basic_indices = self.tracklist[:self.m]
        basics = self.tableau[1:,0]
        
        nonbasic_indices =  self.tracklist[self.m:]
        nonbasics = np.zeros(len(nonbasic_indices))
        #non_basics = en()
        np.set_printoptions(suppress=True)
        return (objective_value, 
                dict(zip(self.tracklist[:self.m], basics)),
                dict(zip(nonbasic_indices, nonbasics)))
        
        
    
    


c = np.array([3., 2])
b = np.array([2., 5, 7])
A = np.array([[1., -1], [3, 1], [4, 3]])
solv = SimplexSolver(c, A, b)


solution = solv.solve()
print(solution)
