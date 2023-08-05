#generators and other matrices for QRM codes
#Praveen Jayakumar, July 2023

import numpy as np
from qrm_utils import filter_wt, leading_bit_index, get_eval_set, min_set, puncture_matrix, puncture_row

def apply_qubit_partition(i, m, qubit_list):
    #applies Plotkin-i partition of passed qubit_list
    #0 - normal Plotkin partition.
    assert i < m, 'Invalid partition specification.'

    row = get_QRM_generators_r1r2(1, 1, m)[m-i-1]
    q1 = []
    q2 = []
    for j, q in enumerate(row):
        if q == 0:
            q1.append(qubit_list[j])
        else:
            q2.append(qubit_list[j])
    return q1, q2

def apply_punc_qubit_partition(i, m, qubit_list, punc_bit_list = [0]):
    #applies Plotkin-i partition of passed qubit_list
    assert i < m, 'Invalid partition specification.'
    
    row = get_QRM_generators_r1r2(1, 1, m)[m-i-1]
    p_row = puncture_row(row, punc_bit_list)

    q1 = []
    q2 = []
    for j, q in enumerate(p_row):
        if q == 0:
            q1.append(qubit_list[j])
        else:
            q2.append(qubit_list[j])
    return q1, q2

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

def get_QRM_punc_generator(C1_params: tuple, C2_params: tuple):
    G2perp, G1q = get_QRM_generator(C1_params=C1_params, C2_params=C2_params)
    G2perpt = puncture_matrix(G2perp)
    G1qp = puncture_matrix(G1q)
    G2perpp = []
    for row in G2perpt:
        if np.sum(row) != len(row):
            G2perpp.append(row)
        else:
            G1qp.append(row)
    return (G2perpp, G1qp)

def get_QRM_generators_r1r2(r1, r2, m):
    #assumes r2>=r1
    if r2 < r1:
        print('Invalid parameters r1: {}, r2: {}, m: {}'.format(r1, r2, m))
    if m == 0:
        return [[1]]
    G1 = Grm(r2, m)
    G1q = filter_wt(G1, weights = [2**(m-i) for i in range(r1, r2 + 1)])
    return G1q

def get_R(G):
    '''
    Row transform matrix for G(r, m)
    
    '''
    k = len(G)
    R = []
    if k == 0:
        return [[]]
    indexes = [leading_bit_index(row) for row in G]
    eval_sets = [get_eval_set(row, reversed=True) for row in G] #reversed since the rows of G are sorted for increasing leading bit position.
    r = max([len(s) for s in eval_sets])
    for i, se in enumerate(eval_sets):
        ms = min_set(se, r, as_int =True)
        js = [0]*k
        for j, s in enumerate(indexes):
            if s in ms:
                js[j] = 1
        R.append(js)
    return R