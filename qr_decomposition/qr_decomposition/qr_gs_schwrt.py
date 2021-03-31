#-----------------------------------------------------------------------------------
#   QR Factorization Using Modified Gram-Schmidt (by Schwarz-Rutishauser) v.0.0.1
#
#        Q,R = qr_gram_schmidt_modsr(A)
#
#        The worst-case complexity of full matrix Q factorization:
#
#           In real & complex cases: p = O(MNlog2(4N^3)
#
#                   An Example: M = 10^3, N = 10^2, p = 2.19 x 1e+6
#
#   GNU Public License (C) 2021 Arthur V. Ratz
#-----------------------------------------------------------------------------------

import numpy as np
import numpy.linalg as lin

def qr_gs_modsr(A, type=complex):
    
    A = np.array(A, dtype=type)
    
    (m,n) = np.shape(A) # Get matrix A's shape m - # of rows, m - # of columns
   
    # Q - an orthogonal matrix of m-column vectors
    # R - an upper triangular matrix (the Gaussian elimination of A to the row-echelon form)
    
    # Initialization: [ Q - multivector Q = A of shape (m x n) ]
    #                 [ R - multivector of shape (n x n)       ]

    Q = np.array(A, dtype=type)      # Q - matrix A
    R = np.zeros((n, n), dtype=type) # R - matrix of 0's    

    # **** Objective: ****

    # For each column vector r[k] in R:
       # Compute r[k,i] element in R, k-th column q[k] in Q;

    for k in range(n):
        # For a span of the previous column vectors q[0..k] in Q, 
        # compute the R[i,k] element in R as the inner product of vectors q[i] and q[k],
        # compute k-th column vector q[k] as the product of scalar R[i,k] and i-th vector q[i],
        # subtracting it from the k-th column vector q[k] in Q
        for i in range(k):

            # **** Compute k-th column q[k] of Q and k-th row r[k] of R **** 
            R[i,k] = np.transpose(Q[:,i]).dot(Q[:,k]);
            Q[:,k] = Q[:,k] - R[i,k] * Q[:,i];
            
        # Compute the r[k,k] pseudo-diagonal element in R 
        # as the Euclidean norm of the k-th vector q[k] in Q,

        # Normalize the k-th vector q[k] in Q, dividing it by the norm r[k,k]
        R[k,k] = lin.norm(Q[:,k]); Q[:,k] = Q[:,k] / R[k,k];
    
    return -Q, -R  # Return the resultant negative matrices Q and R 
