#generators and other matrices
#Praveen Jayakumar, July 2023

import numpy as np
from qrm_utils import *

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

def get_QRM_generator(C1_params: tuple, C2_params: tuple):
    #assumes C2\perp \in C1, returns tuple (Gperp, G\Gperp)
    r1, m1 = C1_params
    r2, m2 = C2_params

    if m1 != m2:
        print('Incompatible code parameters m1, m2: {}, {}'.format(m1, m2))
        return
    if m2 - r2 - 1 > r1:
        print('Incompatible code parameters r1, r2: {}, {}'.format(r1, r2))
        return
    
    G2perp = Hrm(r2, m2)
    G1 = Grm(r1, m1)
    G1q = filter_wt(G1, weights = [2**(i) for i in range(m1 - r1, r2 + 1)])
    return (G2perp, G1q)

def get_R(G):
    '''
    Row transform matrix for G(r, m)
    
    '''
    k = len(G)
    R = []
    indexes = [leading_bit_index(row) for row in G]
    eval_sets = [get_eval_set(row) for row in G]
    r = max([len(s) for s in eval_sets])
    for i, se in enumerate(eval_sets):
        ms = min_set(se, r, as_int =True)
        js = [0]*k
        for j, s in enumerate(indexes):
            if s in ms:
                js[j] = 1
        R.append(js)
    return R
