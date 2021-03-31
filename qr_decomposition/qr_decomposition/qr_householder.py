#----------------------------------------------------------------------------
#   QR Factorization Using Householder Reflections v.0.0.1
#
#        Q,R = qr_householder(A)
#
#        The worst-case complexity of full matrix Q factorization:
#
#              In real case:   p = O(7MN^2 + 4N)
#              In complex case p = O(7MN^2 + 2(4MN + N^2))
#
#              An Example: M = 10^3, N = 10^2, p ~= 7.0 x 1e+9  (real)
#                          M = 10^3, N = 10^2, p ~= 7.82 x 1e+9 (complex)
#
#   GNU Public License (C) 2021 Arthur V. Ratz
#----------------------------------------------------------------------------

import numpy as np
import numpy.linalg as lin

def qr_hh(A, type=complex):
    
    (m,n) = np.shape(A) # Get matrix A's shape m - # of rows, m - # of columns
    
    cols = max(m, n) if m > n else min(m, n) # Determine the A's leading dimension:
    
                                             # Since we want Q - a square (cols x cols), 
                                             # and R - a rectangular matrix (m x n), the dimension must satisfy 
                                             # the following condition:
            
                                             # if m > n, then take a maximum value in (m,n) tuple
                                             # if m <= n, then take a minimum value in (m,n) tuple
    
    # Q - an orthogonal matrix of m-column vectors
    # R - an upper triangular matrix (the Gaussian elimination of A to the row-echelon form)
    
    # Initialization: [ R - a given real or complex matrix A (R = A)                    ]
    #                 [ E - matrix of unit vectors (m x n), I - identity matrix (m x m) ]
    #                 [ Q - orthogonal matrix of basis unit vectors (cols x cols)       ]
    
    E = np.eye(m, n, dtype=type)
    I = np.eye(m, m, dtype=type)
    
    Q = np.eye(cols, cols, dtype=type)
    R = np.array(A, dtype=type)
  
    # **** Objective ****:

    # For each column vector r[k] in R:
       # Compute the Householder matrix H (the rotation matrix); 
       # Reflect each vector in A through its corresponding hyperplane to get Q and R (in-place);
    
    # Take the minimum in tuple (m,n) as the leading matrix A's dimension 
    
    for k in range(min(m, n)):
        
        nsigx = -np.sign(R[k,k])                   # Get the sign of k-th diagonal element r[k,k] in R
        r_norm = lin.norm(R[k:m,k])                # Get the Euclidean norm of the k-th column vector r[k]
        
        u = R[k:m,k] - nsigx * r_norm * E[k:m,k]   # Compute the r[k]'s surface normal u as k-th hyperplane
                                                   # r_norm * E[k:m,k] subtracted from the k-th column vector r[k]

        v = u / lin.norm(u);                       # Compute v, |v| = 1 as the vector u divided by its norm
        
        w = 1.0                                    # Define scalar w, which is initially equal to 1.0
        
        # In complex case:
        
        if type == complex:
            
            # **** Compute the complex Householder matrix H ****
           
            v_hh = np.conj(v)                               # Take the vector v's complex conjugate
            r_hh = np.conj(R[k:m,k])                        # Take the vector r[k]'s complex conjugate
            
            w = r_hh.dot(v) / v_hh.dot(R[k:m,k])            # Get the scalar w as the division of vector 
                                                            # products r_hh and v, v_hh and r[k], respectively

            # **** Compute the real Householder matrix H ****
                
            H = I[k:m,k:m] - (1.0 + w) * np.outer(v, v_hh)  # Compute the Householder matrix as the outer product of
                                                            # v and its complex conjugate multiplied by scalar 1.0 + w ,
                                                            # subtracted from the identity matrix I of shape (k..m x k..m)
            
            # **** Compute the complex orthogonal matrix Q ****
            
            Q_k = np.eye(cols, cols, dtype=type)            # Define the k-th square matrix Q_k of shape 
                                                            # (cols x cols) as an identity matrix
          
            Q_k[k:cols,k:cols] = H;                         # Assign the matrix H to the minor Q_k of shape 
            Q = Q.dot(np.conj(np.transpose(Q_k)))           # [k..cols x k..cols], reducing the minor Q_k's 
                                                            # transpose conjugate to the resultant matrix Q

        # In real case:            
            
        else: 
            # **** Compute the real Householder matrix H ****            
            
            H = I[k:m,k:m] - (1.0 + w) * np.outer(v, v) # Compute the matrix H as the outer product of vector v, 
                                                        # multiplied by the scalar (1.0 + w), subtracting it
                                                        # from the identity matrix I of shape (k..m x k..m)

            # **** Compute the real orthogonal matrix Q ****    
                
            Q_kt = np.transpose(Q[k:cols,:])        # Get the new matrix Q = (Q'[k]^T*H) of shape (m-1 x n) as
            Q[k:cols,:] = np.transpose(Q_kt.dot(H)) # the product of the minor's Q'[k]-transpose and matrix H,
                                                    # where minor Q'[k] - a slice of matrix Q starting at the k-th row
                                                    # (elements of each column vector q[k] in Q, followed by the k-th element)
                                                    # Finally take the new matrix Q's-transpose

        # **** Compute the upper triangular matrix R ****
                        
        R[k:m,k:n] = H.dot(R[k:m,k:n])              # Get the new matrix R of shape (m-1 x n-1) as 
                                                    # the product of matrix H and the minor M'[k,k] <- R[k:m,k:n]
            
    if type != complex:
        Q = np.transpose(Q)                         # If not complex, get the Q's-transpose
        
    return Q, R                                     # Return the resultant matrices Q and R