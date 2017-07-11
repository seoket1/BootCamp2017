# svd_image_compression.py
"""Volume 1A: SVD and Image Compression.
<Name> Eun-Seok Lee
<Class> OSM Lab
<Date> Due 7/11/2017
"""

from scipy import linalg as la
import numpy as np
from matplotlib import pyplot as plt
import cmath

# Problem 1
print("\n\n --Problem 1--\n")

def truncated_svd(A,k=None):
    """Computes the truncated SVD of A. If r is None or equals the number
        of nonzero singular values, it is the compact SVD.
    Parameters:
        A: the matrix
        k: the number of singular values to use
    Returns:
        U - the matrix U in the SVD
        s - the diagonals of Sigma in the SVD
        Vh - the matrix V^H in the SVD
    """
    m, n = np.shape(A)
    AH_A = A.conj().T@A
    eigvalue, eigvector = la.eig(AH_A)
    
    # get the singular values
    singular_values = []
    for i in range(len(eigvalue)):
        singular_values.append(cmath.sqrt(eigvalue[i]))
    
    # ordering eigen vector
    index = np.argsort(singular_values)
    
    eigvec = np.take(eigvector[0,:], index)
    for i in range(1, n):
        temp = np.take(eigvector[i,:], index)
        eigvec = np.vstack((eigvec,temp))
        
    # ordering singular values and make V
    singular_values = np.sort(singular_values)
    singular_values = singular_values[::-1]
    V = eigvec[::-1]
    
    # to make U from A and V
    U = (1/singular_values[0]) * A @ V[:,0]     
    for i in range(1, n):
        U = np.column_stack( (U, (1/singular_values[i]) * A @ V[:,i]) )
    
    # for compact SVD and truncated SVD
    if k == None:
        for i in range(n, m):
            U = np.column_stack((U, np.zeros((m,1))))
        
        s = np.zeros((m, n), dtype = complex)

        for i in range(len(singular_values)):
            s[i,i] = singular_values[i]
    else:
        U = U[:,:k]
        s = np.zeros((k, k), dtype = complex)
        singular_values = singular_values[:k]
        for i in range(len(singular_values)):
            s[i,i] = singular_values[i]     
        V = V[:,:k]                          
    
    
    return U, s, V.conj().T
    raise NotImplementedError("truncated_svd incomplete")

m = 6
n = 4

A = np.random.random((m,n))
b = np.random.random((m,1))

U, s, V_H = truncated_svd(A, 4)

print(np.allclose(A, U@s@V_H))

''' For my practice
keys = [3,1,2]
values = [ "apple", "grape", "peach"]
 
print ("Before sorting")
print (keys)
print (values)

index = np.argsort(keys)
print ("\nindex = " , index)
values = np.take(values, index)
print (values)
keys = np.sort(keys)
 
print ("\nAfter sorting")
print (keys[::-1])
print (values[::-1])
'''

# Problem 2
print("\n\n --Problem 2--\n")

def visualize_svd():
    """Plot each transformation associated with the SVD of A."""
    A2 = np.array([[3, 1],[1, 3]])
    e1 = np.array([[0, 1],[0, 0]])
    e2 = np.array([[0, 0],[0, 1]])   
    U2, s2, V_H2 = truncated_svd(A2) 
    s2@V_H2
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    
    plt.subplot(221)
    plt.axis("equal")
    plt.plot(x, y)
    plt.plot(e1[0], e1[1])
    plt.plot(e2[0], e2[1])
    
    plt.subplot(222)
    plt.axis("equal")
    plt.plot(V_H2[0,0]*np.cos(theta) + V_H2[0,1]*np.sin(theta),\
             V_H2[1,0]*np.cos(theta) + V_H2[1,1]*np.sin(theta))
    plt.plot((V_H2@e1)[0,:], (V_H2@e1)[1,:])
    plt.plot((V_H2@e2)[0,:], (V_H2@e2)[1,:])

    plt.subplot(223)    
    plt.axis("equal")
    plt.plot((s2@V_H2)[0,0]*np.cos(theta) + (s2@V_H2)[0,1]*np.sin(theta),\
             (s2@V_H2)[1,0]*np.cos(theta) + (s2@V_H2)[1,1]*np.sin(theta))
    plt.plot(((s2@V_H2)@e1)[0,:], ((s2@V_H2)@e1)[1,:])
    plt.plot(((s2@V_H2)@e2)[0,:], ((s2@V_H2)@e2)[1,:])
    
    plt.subplot(224)
    plt.axis("equal")
    plt.plot((U2@s2@V_H2)[0,0]*np.cos(theta) + (U2@s2@V_H2)[0,1]*np.sin(theta),\
             (U2@s2@V_H2)[1,0]*np.cos(theta) + (U2@s2@V_H2)[1,1]*np.sin(theta))
    plt.plot((U2@s2@V_H2@e1)[0,:], (U2@s2@V_H2@e1)[1,:])
    plt.plot((U2@s2@V_H2@e2)[0,:], (U2@s2@V_H2@e2)[1,:])
    plt.show()
    #plt.plot(s2@e1, s2@e2)
    
    return U2, s2, V_H2, x, y, e1, e2, 
    raise NotImplementedError("visualize_svd incomplete")


visualize_svd()



# Problem 3
print("\n\n --Problem 3--\n")
def svd_approx(A, k):
    """Returns best rank k approximation to A with respect to the induced 2-norm.

    Inputs:
    A - np.ndarray of size mxn
    k - rank

    Return:
    Ahat - the best rank k approximation
    """
    U,s,Vh = la.svd(A, full_matrices=False)
    S = np.diag(s[:k])
    Ahat = U[:,:k].dot(S).dot(Vh[:k,:])
    norm = la.norm(A-Ahat)
    return Ahat, norm
    raise NotImplementedError("svd_approx incomplete")
    
A3 = np.array([[1,1,3,4], [5,4,3,7], [9,10,10,12], [13,14,15,16], [17,18,19,20]])
A3hat, A3_norm = svd_approx(A3, 3)
print("Please see the codes.\n")


# Problem 4
print("\n\n --Problem 4--\n")
def lowest_rank_approx(A,e):
    """Returns the lowest rank approximation of A with error less than e
    with respect to the induced 2-norm.

    Inputs:
    A - np.ndarray of size mxn
    e - error

    Return:
    Ahat - the lowest rank approximation of A with error less than e.
    """
    U,s,Vh = la.svd(A, full_matrices=False)
    
    for i in range(len(s) - 1, -1, -1):
        if s[i] > e:
            lowest_rank = i + 1
            break
        
    Ahat, norm = svd_approx(A, lowest_rank)    
    return Ahat, norm, lowest_rank
    raise NotImplementedError("lowest_rank_approx incomplete")

m = 6
n = 4

A5 = np.random.random((m,n))
Ahat4, norm4, lowest_rank = lowest_rank_approx(A5, 0.3)
print("Please see the codes.\n")


# Problem 5
print("\n\n --Problem 5--\n")
def compress_image(img, k):
    """Plot the original image found at 'filename' and the rank k approximation
    of the image found at 'filename.'

    filename - jpg image file path
    k - rank
    """
    R = plt.imread(img)[:,:,0].astype(float) /256
    G = plt.imread(img)[:,:,1].astype(float) /256
    B = plt.imread(img)[:,:,2].astype(float) /256
    
    Rk, Rk_norm = svd_approx(R, k)
    Gk, Gk_norm = svd_approx(G, k)
    Bk, Bk_norm = svd_approx(B, k)
    
    
    Rk[Rk > 1] = 1
    Gk[Gk > 1] = 1
    Bk[Bk > 1] = 1
    
    Rk[Rk < 0] = 0
    Gk[Gk < 0] = 0
    Bk[Bk < 0] = 0

    new_image = np.dstack((Rk,Gk,Bk))
    return new_image
    raise NotImplementedError("compress_image incomplete")
# Take only one layer (layer 0) of the image


original_img = compress_image('C:/Users/suket/Desktop/Homeworks/Computation/Week3/Problem3/hubble.jpg', None)
new_img = compress_image('C:/Users/suket/Desktop/Homeworks/Computation/Week3/Problem3/hubble.jpg', 20)

plt.subplot(121)
plt.title("Original Image")
plt.imshow(original_img)
plt.subplot(122)
plt.title("Rank 20 Approximation")
plt.imshow(new_img)
plt.show()  
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
