#======================================================================
#
#     This routine interfaces with the TASMANIAN Sparse grid
#     The crucial part is 
#
#     aVals[iI]=solveriter.iterate(aPoints[iI], n_agents)[0]  
#     => at every gridpoint, we solve an optimization problem
#
#     Simon Scheidegger, 11/16 ; 07/17
#======================================================================

import TasmanianSG
import numpy as np
from parameters import *
import nonlinear_solver_iterate as solveriter
'''
#======================================================================
def sparse_grid_iter(n_agents, iDepth, valold):
    
    grid  = TasmanianSG.TasmanianSparseGrid()

    k_range=np.array([k_bar, k_up])

    ranges=np.empty((n_agents, 2))


    for i in range(n_agents):
        ranges[i]=k_range

    iDim=n_agents
    iOut=1

    grid.makeLocalPolynomialGrid(iDim, iOut, iDepth, which_basis, "localp")
    grid.setDomainTransform(ranges)

    aPoints=grid.getPoints()
    iNumP1=aPoints.shape[0]
    aVals=np.empty([iNumP1, 1])
    
    file=open("comparison1.txt", 'w')
    for iI in range(iNumP1):
        aVals[iI]=solveriter.iterate(aPoints[iI], n_agents, valold)[0]
        v=aVals[iI]*np.ones((1,1))
        to_print=np.hstack((aPoints[iI].reshape(1,n_agents), v))
        np.savetxt(file, to_print, fmt='%2.16f')
        
    file.close()
    grid.loadNeededPoints(aVals)
    
    f=open("grid_iter.txt", 'w')
    np.savetxt(f, aPoints, fmt='% 2.16f')
    f.close()
    
    return grid


'''
def sparse_grid_iter(n_agents, iDepth, valold, refinement_level, count):

    
    fTol = 1.E-5
    
    
    
    grid1  = TasmanianSG.TasmanianSparseGrid()
    grid2 = TasmanianSG.TasmanianSparseGrid()
    
    
    k_range=np.array([k_bar, k_up])

    ranges=np.empty((n_agents, 2))


    for i in range(n_agents):
        ranges[i]=k_range

    iDim=n_agents
    iOut=1
    
    # level of grid before refinement
    grid1.makeLocalPolynomialGrid(iDim, iOut, iDepth, which_basis, "localp")
    grid1.setDomainTransform(ranges)
    
    aPoints=grid1.getPoints()
    iNumP1=aPoints.shape[0]
    aVals=np.empty([iNumP1, 1])
    
    for iI in range(iNumP1):
        aVals[iI]=solveriter.iterate(aPoints[iI], n_agents, valold, count)[0]
        v=aVals[iI]*np.ones((1,1))
    grid1.loadNeededPoints(aVals)
    
    #refinement level
    for iK in range(refinement_level):
        grid1.setSurplusRefinement(fTol, 1, "fds")   #also use fds, or other rules
        aPoints = grid1.getNeededPoints()
        aVals = np.empty([aPoints.shape[0], 1])
        for iI in range(aPoints.shape[0]):
            aVals[iI]=solveriter.iterate(aPoints[iI], n_agents, valold, count)[0]
            v=aVals[iI]*np.ones((1,1))
        grid1.loadNeededPoints(aVals)
    
    grid2 = TasmanianSG.TasmanianSparseGrid()
    grid2.makeLocalPolynomialGrid(iDim, iOut, refinement_level+iDepth, which_basis, "localp")
    grid2.setDomainTransform(ranges)
    a = grid2.getNumPoints()
 
    print("\n-------------------------------------------------------------------------------------------------")
    print ("   a fix sparse grid of level ", refinement_level+iDepth, " would consist of " ,a, " points")
    print("\n-----------------------------------     iter   ------------------------------------------------\n")    
    print(count)
    
    return grid1

#======================================================================
