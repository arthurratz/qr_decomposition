#---------------------------------------------------------------------------
#   QR Factorization Using Gram-Schmidt Orthogonalization v.0.0.1
#
#        Q,R = qr_gram_schmidt(A)
#        
#        The worst-case complexity of full matrix Q factorization:
#
#              In real case:   p = O(5MNlog2(N))
#              In complex case p = O(5MNlog2(2N) + 6MN)
#
#              An Example: M = 10^3, N = 10^2, p ~= 3.33 x 1e+10 (real)
#                          M = 10^3, N = 10^2, p ~= 3.82 x 1e+10 (complex)
#
#   GNU Public License (C) 2021 Arthur V. Ratz
#---------------------------------------------------------------------------

import numpy as np
import numpy.linalg as lin

# Function: conj_transpose(a) returns the conjugate transpose of vector b

def conj_transpose(a): # Takes the conjugate of complex vector a's-transpose
    return np.conj(np.transpose(a))

# Function: proj_ab(a,b) returns the projection of vector a onto vector b

def proj_ab(a,b): # Divides the product of <a,b> by the norm |b| and multiply the scalar by vector b:
                  # In real case:     <a,b> - the inner product of two real vectors
                  # In complex case:  <a,b> - the inner product of vector a and vector b's conjugate transpose
    return a.dot(conj_transpose(b)) / lin.norm(b) * b

def qr_gs(A, type=complex):
    
    A = np.array(A, dtype=type)
    
    (m, n) = np.shape(A) # Get matrix A's shape m - # of rows, m - # of columns
    
    (m, n) = (m, n) if m > n else (m, m)    # Determine the A's leading dimension:
    
                                            # The leading dimension must satisfy the following condition:
                                            #
                                            # if m > n, then the shape remains unchanged,
                                            #           and Q,R - are rectangular matrices (m x n)
                                            #
                                            # if m <= n, then the the shape of Q,R is changed to (m x m),
                                            #            and Q,R - are square matrices
    
    # Q - an orthogonal matrix of m-column vectors
    # R - an upper triangular matrix (the Gaussian elimination of A to the row-echelon form)
    
    # Initialization: [ Q - a multivector of shape (m x n) ] 
    #                 [ R - a multivector of shape (m x n) ]

    Q = np.zeros((m, n), dtype=type) # Q - matrix of 0's
    R = np.zeros((m, n), dtype=type) # R - matrix of 0's
    
    # **** Objective: ****
    
    # For each column vector a[k] in A:
       # Compute the sum of a[k]'s projections onto a span of column vectors a[0..k]
       # Compute all column vectors q[k] of the orthogonal matrix Q
    
    # Finally: Compute the upper triangular matrix R of matrix A
    
    # Take n as the leading matrix A's dimension 
    
    for k in range(n):
        
        pr_sum = np.zeros(m, dtype=type)       # Define the sum of projections vector pr_sum
        
        # For a span of column vectors q[0..k] in Q, 
        # compute the sum f a[k]'s projections onto each vector q[0..k] in Q
        
        for j in range(k):
            pr_sum += proj_ab(A[:,k], Q[:,j])  # Get the projection of a[k] onto q[j],
                                               # and accumulate the result to the vector pr_sum

        # **** Compute the orthogonal matrix Q ****                
                
        Q[:,k] = A[:,k] - pr_sum               # Compute the new k-th column vector q[k] in Q by
                                               # subtracting the sum of projections from the column vector a[k]
        
        Q[:,k] = Q[:,k] / lin.norm(Q[:,k])     # Normalize the new k-th column vector q[k] dividing it by its norm

    if type == complex:
        
        # **** Compute the complex upper triangular matrix R ****
        
        R = conj_transpose(Q).dot(A)           # Compute the matrix R as the product of Q's 
                                               # transpose conjugate and matrix A (real case)

        # **** Compute the real upper triangular matrix R ****
    
    else: R = np.transpose(Q).dot(A)           # Compute the matrix R as the product of Q's-transpose 
                                               # and matrix A (complex case)
            
    return -Q, -R                              # Return the resultant negative matrices Q and R