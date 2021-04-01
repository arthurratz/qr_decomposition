#------------------------------------------------------------------------------------------
#   QR Factorization of Matrix A In Python 3.8.1 Numpy 1.19.2
#
#   The following sample demonstrates the QR factorization of 
#   a randomly generated matrix A of real or complex numbers using:
#
#        * Gram-Schmidt Orthogonalization;
#        * Householder Reflections;
#   
#   , and surveys the complexity and performance (single-threaded) of these both methods
#
#   GNU Public License (C) 2021 Arthur V. Ratz
#------------------------------------------------------------------------------------------

import time
import math
import random
import pandas as pd

import numpy as np
import numpy.linalg as lin

from qr_gschmidt import *
from qr_gs_schwrt import *
from qr_householder import *

mat_shape      = { 'min': 3,   'max': 10   }
mat_shape_perf = { 'min': 750, 'max': 950  }

qr_alg         = [ { 'alg': qr_gs,       'name': 'Gram-Schmidt       ' },
                   { 'alg': qr_gs_modsr, 'name': 'Schwarz-Rutishauser' },
                   { 'alg': qr_hh,       'name': 'Householder        ' } ]

mat_types      = [ 'real   ', 'complex' ]

checkup_status = [ 'failed', 'passed' ]

checkup_banner = "\n[ Verification %s... ]"
stats_banner   = "%s Matrix A Statistics:\n"
qr_test_banner = "\nQR Factorization Of A `%s` Matrix Using %s Algorithm:"
survey_banner  = "Matrix: %s    WINS: [ %s : %d secs  ] LOOSES: [ %s : %d secs ]"
perf_stats     = "%s : [ type: `%s` exec_time: %d secs verification: %s ]"

app_banner     = "QR Factorization v.0.0.1 CPOL License (C) 2021 by Arthur V. Ratz"

# Function: perf(A, qr, type=complex) evaluates the qr factorization method's execution wall-time in nanoseconds,
#           returns the tuple of the resultant matrices Q,R and the execution time

def perf(A, qr, type=complex):
    t_d = time.time(); Q,R = qr(A, type); \
        return Q, R, (time.time() - t_d)

def check(M1, M2):
    v1 = np.reshape(M1,-1)
    v2 = np.reshape(M2,-1)
    if len(v1) != len(v2):
       return False
    else: return 0 == len(np.where(np.array(\
       [ format(c1, '.4g') == format(c2, '.4g') \
        for c1,c2 in zip(v1, v2) ]) == False)[0])

def rand_matrix(rows, cols, type=complex):
    np.set_printoptions(precision=8)
    if type == complex:
        return np.reshape(\
              np.random.uniform(1, 10, rows * cols) + \
              np.random.uniform(-10, 10, rows * cols) *  1j, (rows, cols))
    else: return np.reshape(10 * np.random.uniform(\
            0.01, 0.99, rows * cols), (rows, cols))
    
def print_matrix(M, alias):
    np.set_printoptions(\
        precision=2, suppress=True, \
        formatter='complexfloat')
    if isinstance(M, complex):
        eps = np.finfo(float).eps; tol = 100
        M = [np.real(m) if np.imag(m)<tol*eps else m for m in M]
        M = [np.asscalar(np.real_if_close(m)) for m in M]
    print("\nMatrix %s (%dx%d):" % \
        (alias, len(M), len(M[0])),"\n")
    pd.set_option('precision', 2); \
        df = pd.DataFrame(M)
    df = df.to_string(index=False).replace('j','i')
    print(df)

def logo():
    print(app_banner)
    print(''.join(['=' for p in range(len(app_banner))]))
    
def qr_demo(s, qr, type=complex):

    print(qr_test_banner % (\
        "Complex" if type == complex \
            else "Real", s.replace(' ', '')))
    
    rows = np.random.randint(\
        mat_shape['min'], mat_shape['max'])

    cols = np.random.randint(\
        mat_shape['min'], mat_shape['max'])
    
    A = rand_matrix(rows, cols, type); print_matrix(A, "A")
    
    Q,R,T = perf(A, qr, type)
        
    status = check(A, Q.dot(R))

    A = np.around(A, decimals=2)
    Q = np.around(Q, decimals=2)
    R = np.around(R, decimals=2)

    print_matrix(Q, "Q")
    print_matrix(R, "R")
    
    print(checkup_banner % (checkup_status[status]),"\n")

    return status

def qr_perf():
    
    print ("\nPerformance Assessment:")
    print ("=======================================")
    
    rows = np.random.randint(\
        mat_shape_perf['min'], mat_shape_perf['max'])

    cols = np.random.randint(\
        mat_shape_perf['min'], mat_shape_perf['max'])
    
    d = np.random.randint(5)
    cols += d if random.uniform(0,1) > 0.5 else -d
    
    A = [ rand_matrix(rows, cols, float), \
          rand_matrix(rows, cols, complex) ]
    
    print ("\nMatrix A (%d x %d):" % (rows, cols))
    print ("============================\n")
    
    exec_time = np.zeros((len(mat_types), len(qr_alg)))
    survey = np.zeros((len(mat_types), 1), dtype=object)
    
    status = 0
    for s_j in range(len(mat_types)):
        for s_i in range(len(qr_alg)):
            qr, name = \
                qr_alg[s_i]['alg'], qr_alg[s_i]['name']; \
                Q, R, exec_time[s_j][s_i] = perf(A[s_j],   \
                    qr, float if s_j % len(mat_types) == 0 else complex); \
                status = checkup_status[check(A[s_j], Q.dot(R))]
            print(perf_stats % (name, mat_types[s_j].replace(' ','').lower(), \
                exec_time[s_j][s_i], status))
        
            if status == "failed": break

        if status == "failed":
           print_matrix(A[s_j], "A")
           print_matrix(Q, "Q")
           print_matrix(R, "R")
            
           print("\n*** FAILURE!!! ****\n")
        
           break
            
        wr_time, lr_time = \
            np.min(exec_time[s_j]), \
            np.max(exec_time[s_j])
    
        wi = np.where(exec_time[s_j] == wr_time)[0][0]
        li = np.where(exec_time[s_j] == lr_time)[0][0]
    
        s_w = qr_alg[wi]['name']; s_l = qr_alg[li]['name']
            
        survey[s_j] = { 'alg_w': s_w, 'tm_w': wr_time, \
                        'alg_l': s_l, 'tm_l': lr_time }

        print("\n")
        
    for s_j in range(len(mat_types)):
        print(survey_banner % (mat_types[s_j].lower(),
              survey[s_j][0]['alg_w'], survey[s_j][0]['tm_w'], \
              survey[s_j][0]['alg_l'], survey[s_j][0]['tm_l']))
    
def main():
    
    logo()
    
    np.random.seed(int(time.time()))
    
    status = 0
    for s_j in range(len(qr_alg)):
        for s_i in range(len(mat_types)):
            status = qr_demo(qr_alg[s_j]['name'], \
                qr_alg[s_j]['alg'], complex if s_i % 2 else float)
            if status == False: break

        if status == False:
           print("\n*** FAILURE!!! ****\n"); break

    qr_perf(); print("\n")

if __name__ == "__main__":
    main()
