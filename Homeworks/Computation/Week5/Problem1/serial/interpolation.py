#======================================================================
#
#     This routine interfaces with the TASMANIAN Sparse grid
#     The crucial part is 
#
#     aVals[iI]=solver.initial(aPoints[iI], n_agents)[0]  
#     => at every gridpoint, we solve an optimization problem
#
#     Simon Scheidegger, 11/16 ; 07/17
#======================================================================

import TasmanianSG
import numpy as np
from parameters import *
import nonlinear_solver_initial as solver

#======================================================================
'''
def sparse_grid(n_agents, iDepth):
    
    grid  = TasmanianSG.TasmanianSparseGrid()

    k_range=np.array([k_bar, k_up])

    ranges=np.empty((n_agents, 2))


    for i in range(n_agents):
        ranges[i]=k_range

    iDim=n_agents

    grid.makeLocalPolynomialGrid(iDim, iOut, iDepth, which_basis, "localp")
    grid.setDomainTransform(ranges)

    aPoints=grid.getPoints()
    iNumP1=aPoints.shape[0]
    aVals=np.empty([iNumP1, 1])
    
    file=open("comparison0.txt", 'w')
    for iI in range(iNumP1):
        aVals[iI]=solver.initial(aPoints[iI], n_agents)[0] 
        v=aVals[iI]*np.ones((1,1))
        to_print=np.hstack((aPoints[iI].reshape(1,n_agents), v))
        np.savetxt(file, to_print, fmt='%2.16f')
        
    file.close()
    grid.loadNeededPoints(aVals)
    
    f=open("grid.txt", 'w')
    np.savetxt(f, aPoints, fmt='% 2.16f')
    f.close()
    print("gridgridgridgridgridgridgridgrid")
    print(grid)
    return grid



'''
def sparse_grid(n_agents, iDepth, refinement_level):
    fTol = 1.E-5
    
    
      
    grid  = TasmanianSG.TasmanianSparseGrid()
    grid2 = TasmanianSG.TasmanianSparseGrid()

    k_range=np.array([k_bar, k_up])

    ranges=np.empty((n_agents, 2))


    for i in range(n_agents):
        ranges[i]=k_range

    iDim=n_agents

    # level of grid before refinement
    grid.makeLocalPolynomialGrid(iDim, iOut, iDepth, which_basis, "localp")
    grid.setDomainTransform(ranges)

    aPoints=grid.getPoints()
    iNumP1=aPoints.shape[0]
    aVals=np.empty([iNumP1, 1])
    
    file=open("comparison0.txt", 'w')
    for iI in range(iNumP1):
        aVals[iI]=solver.initial(aPoints[iI], n_agents)[0] 
        v=aVals[iI]*np.ones((1,1))
        to_print=np.hstack((aPoints[iI].reshape(1,n_agents), v))
        np.savetxt(file, to_print, fmt='%2.16f')
        
    file.close()
    grid.loadNeededPoints(aVals)
    
    #refinement level
    for iK in range(refinement_level):
        grid.setSurplusRefinement(fTol, 1, "fds")   #also use fds, or other rules
        aPoints = grid.getNeededPoints()
        aVals = np.empty([aPoints.shape[0], 1])
        for iI in range(aPoints.shape[0]):
            aVals[iI]=solver.initial(aPoints[iI], n_agents)[0] 
            v=aVals[iI]*np.ones((1,1))
        grid.loadNeededPoints(aVals)
    
    grid2 = TasmanianSG.TasmanianSparseGrid()
    grid2.makeLocalPolynomialGrid(iDim, iOut, refinement_level+iDepth, which_basis, "localp")
    grid2.setDomainTransform(ranges)
    a = grid2.getNumPoints()
 
    print("\n-------------------------------------------------------------------------------------------------")
    print ("   a fix sparse grid of level ", refinement_level+iDepth, " would consist of " ,a, " points")
    print("\n-------------------------------------   init     -----------------------------------------------\n")    

    
    return grid

#======================================================================

