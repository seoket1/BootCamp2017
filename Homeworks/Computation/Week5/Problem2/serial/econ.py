#======================================================================
# 
#     sets the economic functions for the "Growth Model", i.e., 
#     the production function, the utility function
#     
#
#     Simon Scheidegger, 11/16 ; 07/17
#====================================================================== 

from parameters import *
import numpy as np


#====================================================================== 
#utility function u(c,l) 

def utility(cons=[], lab=[]):
    sum_util=0.0
    n=len(cons)
    for i in range(n):
        nom1=(cons[i]/big_A)**(1.0-gamma) -1.0
        den1=1.0-gamma
        
        nom2=(1.0-psi)*((lab[i]**(1.0+eta)) -1.0)
        den2=1.0+eta
        
        sum_util+=(nom1/den1 - nom2/den2)
    
    util=sum_util
    #print(util)
    return util 


#====================================================================== 
# output_f 

def output_f(kap=[], lab=[]):
    fun_val = big_A*(kap**psi)*(lab**(1.0 - psi))  #np.array([0.9, 0.95, 1.00, 1.05, 1.10]).T * 
    fun_val_z = np.zeros((len(theta),len(kap)))

    for i in range(itheta):
        for j in range(len(kap)):
            fun_val_z[i, j] = theta[i] * fun_val[j]
    #print(fun_val_z)
    return fun_val_z

#======================================================================
