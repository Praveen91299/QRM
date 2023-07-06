#code for QRM gate counts and circuit constructions

import math
import numpy as np
 
def binom_sum(m,start,end):
    if start > end:
        print('Error, start > end')
        return
    if start > m or end > m:
        print('Print comb start or end larger than m')
        return
    return np.sum([math.comb(m, i) for i in range(start, end + 1)])

def naive_CX_count(r, m):
    """
    Gate counts for naive implementations, with no row transforms
    """
    if r>m:
        print('Invalid parameters r = {} > m = {}'.format(r, m))
        return
    if r ==2 and m == 2:
        return 4
    return np.sum([math.comb(m, i)*(2**(m-i) - 1) for i in range(r+1)])

def std_CX_count(r, m):
    """
    Gate counts for standard row transformed G(r, m)
    """
    if r>m:
        print('Invalid parameters r = {} > m = {}'.format(r, m))
        return
    if r == m:
        return 0
    if r == m-r-1:
        return binom_sum(m, 0, m-r-1)*(2**(r+1)-1)
    return binom_sum(m, m-r, r)*(2**(m-r) - 1) + binom_sum(m, 0, m-r-1)*(2**(r+1)-1)#np.sum([math.comb(m, i)*(2**(m-i) - 1) for i in range(r+1)])

def Urr_CX_count(r):
    """
    Gate counts for recursive U_{r,r}
    """
    if r ==1:
        return 1 
    g = 2**(r-1)
    return g + 2*Urr_CX_count(r-1)

def rec_CX_count(r, m):
    """
    Gate counts for recursive U_{r,m}
    """
    if r>m:
        print('Invalid parameters r = {} > m = {}'.format(r, m))
        return
    if r==m:
        return Urr_CX_count(r)
    else:
        return binom_sum(m-1, 0, r) + math.comb(m-1, m-r-1) + 2*rec_CX_count(r, m-1)

def get_full_matrix(m):
    """
    Full tensored matrix [[1, 1], [0, 1]]^\otimes m
    """
    B = [[1, 1], [0, 1]]
    F = [1]
    for _ in range(m):
        F = np.kron(F, B)
    return F

def Hrm(r, m):
    #returns parity check for RM(r, m)
    F = get_full_matrix(m)
    return [F[i] for i in range(len(F)) if sum(F[i]) >=2**(r+1)]

def Grm(r, m):
    #generator for RM(r, m)
    F = get_full_matrix(m)
    return [F[i] for i in range(len(F)) if sum(F[i]) >=2**(m-r)]

def reorder_wt(M, reverse = True):
    #orders rows with weight
    return np.array(sorted(M, key=lambda x: np.sum(x), reverse=reverse))

def Hqrm(r, m):
    '''
    parity check for the QRM CSS code
    returns H with (X|Z) convention
    '''
    H_rm = Hrm(r, m)
    z = np.zeros_like(H_rm)
    Hx = np.concatenate((H_rm, z), axis = 1)
    Hz = np.concatenate((z, H_rm), axis = 1)
    return np.concatenate((Hx, Hz), axis =0)