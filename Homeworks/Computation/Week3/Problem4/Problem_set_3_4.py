import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from scipy.stats import linregress
import math
from scipy import linalg as la
import cmath

# Problem 1
print("\n\n --Problem 1--\n")
def drazin_test(A,Ad, k):
    if np.allclose(A@Ad, Ad@A) and np.allclose(np.linalg.matrix_power(A, k+1)@Ad, np.linalg.matrix_power(A, k)) and np.allclose(Ad@A@Ad, Ad):
        return True
    else:
        return False
    
A = np.array([[1,3,0,0],[0,1,3,0],[0,0,1,3],[0,0,0,0]])
Ad = np.array([[1,-3,9,81],[0,1,-3,-18],[0,0,1,3],[0,0,0,0]])
B = np.array([[1,1,3],[5,2,6],[-2,-1,-3]])
Bd = np.array([[0,0,0],[0,0,0],[0,0,0]])

print(drazin_test(A,Ad,1))
print(drazin_test(B,Bd,3))

# Problem 2
print("\n\n --Problem 2--\n")
def drazin(A, tol):
    n, n = np.shape(A)
    f = lambda x: abs(x) > tol
    Q1, S, k1 = la.schur(A, sort = f)
    g = lambda y: abs(y) <= tol
    Q2, T, k2 = la.schur(A, sort = g)
    U = np.column_stack((S[:,:k1], T[:,:n-k1]))
    U_inverse = la.inv(U)
    V = U_inverse@A@U
    Z = np.zeros((n,n), dtype = float)
    
    if k1 != 0:
        M_inverse = la.inv(V[:k1,:k1])
        Z[:k1,:k1] = M_inverse
    return U@Z@U_inverse
    
    
A = np.array([[1,3,0,0],[0,1,3,0],[0,0,1,3],[0,0,0,0]])    
A_d = drazin(A, 1e-5)
print("\n",A)
print("\n",A_d)
    
# Problem 3
print("\n\n --Problem 3--\n")
from scipy.sparse import csgraph

def effective_resistance(A):
    L = csgraph.laplacian(A)

    m, n = np.shape(L)
    R = np.zeros((m,n))
    
    identity = np.identity(n)

    for i in range(m):
        for j in range(n):
            L_copy = np.copy(L)
            if i != j:
                L_copy[j,:] = identity[j,:]
                R[i, j] = drazin(L_copy, 1e-5)[i, i]
            else:
                R[i, j] = 0
    return R

A3_1 = np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]])
A3_2 = np.array([[0,4],[4,0]])
R3_1 = effective_resistance(A3_1)
R3_2 = effective_resistance(A3_2)
print(R3_1)
print(R3_2)

# Problem 4 and Problem 5
print("\n\n --Problem 4 and Problem 5--\n")

import pandas as pd

class LinkPredictor():
    def __init__(self, csvfile):
        # data arrange. convert to np_data #csvfile, header = -1) # 
        data = pd.read_csv(csvfile, header = -1)
        npdata = data.values
        transition = np.hstack((npdata[:,0], npdata[:,1]))
        
        # create name_list
        name_list = np.unique(transition)
        
        # making index
        index = np.zeros(len(name_list))
        for i in range(len(name_list)):
            index[i] = int(i)
        
        # making ordered data with index
        ordered_data = np.column_stack((name_list.T, index.T))
        
        # making adjacency matrix
        adjacency = np.zeros((len(name_list),len(name_list)))
        temp = []
        for i in range(len(name_list)):
            for j in range(len(npdata[:,0])):
                if name_list[i] == npdata[j, 0]:
                    temp.append(npdata[j, 1])
            for k in range(len(temp)):
                for l in range(len(ordered_data)):
                    if temp[k] == ordered_data[l, 0]:
                        adjacency[i, int(ordered_data[l, 1])] = 1
                        adjacency[int(ordered_data[l, 1]), i] = 1     
            temp = []

        # checking adjaceny matrix
        np.allclose(adjacency, adjacency.T)
        effec_resi = effective_resistance(adjacency)
        
        # save as an attribute
        self.ordered_name_list = ordered_data
        self.adjacency = adjacency
        self.effec_resi = effec_resi
        self.npdata = npdata
        
    def predict_link(self, node = None):
        effec_resi = np.copy(self.effec_resi)
        ordered_data = np.copy(self.ordered_name_list)
        adjacency = np.copy(self.adjacency)
        #node = "Alan"
        adjacency[adjacency == 0] = 10
        adjacency[adjacency == 1] = 0
        effec_resi = effec_resi * adjacency
        effec_resi[effec_resi == 0] = 100
        #loc = np.where("Carol" == ordered_data)
        if node == None:
            minval = np.min(effec_resi)
            loc = np.where(minval == effec_resi)
            return (ordered_data[int(loc[0]),0], ordered_data[int(loc[1]),0])
        else:
            for i in range(len(ordered_data)):
                if node == ordered_data[i,0]:
                    minval2 = np.min(effec_resi[i,:])
                    loc2 = np.where(minval2 == effec_resi)
                    print(loc2)
                    return ordered_data[loc2[1],0]
            raise ValueError('could not find %s' % (node))

    def add_link(self, input1, input2):
        ordered_data = self.ordered_name_list
        if np.any(ordered_data[:,0][input1 == ordered_data[:,0]]) and np.any(ordered_data[:,0][input2 == ordered_data[:,0]]):
            npdata = self.npdata
            npdata = np.vstack((npdata, [input1, input2]))
            transition = np.hstack((npdata[:,0], npdata[:,1]))
        
            # create name_list
            name_list = np.unique(transition)
      
            # making adjacency matrix
            adjacency = np.zeros((len(name_list),len(name_list)))
            temp = []
            for i in range(len(name_list)):
                for j in range(len(npdata[:,0])):
                    if name_list[i] ==  npdata[j, 0]:
                        temp.append(npdata[j, 1])
                for k in range(len(temp)):
                    for l in range(len(ordered_data)):
                        if temp[k] == ordered_data[l, 0]:
                            adjacency[i, int(ordered_data[l, 1])] = 1
                            adjacency[int(ordered_data[l, 1]), i] = 1     
                temp = []
            
            effec_resi = effective_resistance(adjacency)
       
        
            # save as an attribute
            self.ordered_name_list = ordered_data
            self.adjacency = adjacency
            self.effec_resi = effec_resi
            self.npdata = npdata
        else:
            raise ValueError('could not find %s' % (node))
            
social = LinkPredictor("C:/Users/suket/Desktop/Homeworks/Computation/Week3/Problem4/social_network.csv")
print(social.predict_link())
print(social.predict_link("Melanie"))
print(social.predict_link(node = "Alan"))


social.add_link("Alan", "Sonia")
print(social.predict_link("Alan"))

social.add_link("Alan", "Piers")
print(social.predict_link("Alan"))




