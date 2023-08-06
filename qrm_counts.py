#predicted gate counts for various QRM code encoders
#Praveen Jayakumar, July 2023

#need to review counts
from qrm_utils import binom_sum
import math
import numpy as np

def naive_CX_count(r, m):
    if r>m:
        print('Invalid parameters r = {} > m = {}'.format(r, m))
        return
    if r ==2 and m == 2:
        return 4
    return np.sum([math.comb(m, i)*(2**(m-i) - 1) for i in range(r+1)])

def std_CX_count(r, m):
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
    Gate counts for recursive U_rr, quantum or classical
    """
    if r ==1:
        return 1 
    g = 2**(r-1)
    return g + 2*Urr_CX_count(r-1)

def Urm_CX_count(r, m):
    '''
    Similar to Urr_CX_count(), for general G(r, m) encoding, quantum or classical
    '''
    if r == 0:
        return 2**m - 1
    if r == m:
        return Urr_CX_count(m)
    return binom_sum(m-1, 0, r) + 2*Urm_CX_count(r, m-1)  # (u, u) + (0, v) -> (u, 0) + (0, u+v)

def punc_Urm_CX_count(r, m, state_prep = False):
    '''
    punctured, classical code state encoder

    Similar to Urr_CX_count(), for general G(r, m) encoding
    '''
    if state_prep:
        e = -1
    else:
        e = 0
    
    if r == 0:
        return 2**m - 2
    if r + 1 == m:
        if not state_prep:
            return 2**r + punc_Urm_CX_count(r-1, r, state_prep=state_prep) + Urr_CX_count(r)
    if r == m:
        assert state_prep
        if r == 1:
            return 0
        return 2**(r-1) - 1 + punc_Urm_CX_count(r-1, r-1, state_prep=state_prep) + Urr_CX_count(r-1)
    return binom_sum(m-1, 0, r) + e + punc_Urm_CX_count(r, m-1, state_prep=state_prep) + Urm_CX_count(r, m-1)  # (u, u) + (0, v) -> (u, 0) + (0, u+v)


def rec_CX_count(r, m):
    if r>m or r < (m-1)//2:
        print('Invalid parameters r = {}, m = {}'.format(r, m))
        return
    if r==m:
        return Urr_CX_count(r)
    if m-r-1 == r:
        return math.comb(m-1, m-r-1) + 2*rec_CX_count(r, m-1)
    else:
        return binom_sum(m-1, m-r, r) + math.comb(m-1, m-r-1) + 2*rec_CX_count(r, m-1)

def rec_CX_count_assym(r, m, r_in, m_in):
    '''
    gate counts for assymmetric QRM code
    verified July 29
    '''
    if r > m or r < -1:
        print('Invalid parameters r, m : {}, {}'.format(r, m))
        return
    if 2*r_in + 1 > m_in:
        if m == r_in:
            return Urr_CX_count(m)
    if 2*r_in + 1 <=m_in:
        if r == -1:
            return Urm_CX_count(r_in, m)
    return binom_sum(m-1, r, r_in) + 2*rec_CX_count_assym(r-1, m-1, r_in, m_in)

def rec_CX_count_punc(r, m, classical=False, state_prep=False):
    '''
    CNOT gate counts for recursive encoder of punctured QRM(r, m)^*
    Verified July 29
    '''
    if classical:
        return punc_Urm_CX_count(r, m, state_prep=state_prep)
    if r >= m or r < (m-1)//2:
        print('Invalid Parameters r, m: {}, {}\nRequire r < m'.format(r, m))
        return
    if m==1 and r ==0:
        return 0
    if m-r-1 == r:
        return math.comb(m-1, m-r-1) + rec_CX_count_punc(r, m-1, state_prep=state_prep) + rec_CX_count(r, m-1)
    if m == r+1:
        return 2**r + rec_CX_count_punc(r-1, r, state_prep=state_prep) + Urr_CX_count(r)
    return binom_sum(m-1, m-r, r) + math.comb(m-1, m-r-1) + rec_CX_count_punc(r, m-1, state_prep=state_prep) + rec_CX_count(r, m-1)

def rec_CX_count_assym_punc(r, m, r_in, m_in, state_prep = False):
    
    #gate counts for assymmetric QRM code
    if 2*r_in + 1 >= m_in: #quantum state
        if r == 2*r_in - m_in + 1:
            return punc_Urm_CX_count(r_in, r_in + 1, state_prep=state_prep)
    if 2*r_in + 1 < m_in: #classical state
        if r == 0:
            return punc_Urm_CX_count(r_in, m, state_prep=state_prep)
    return binom_sum(m-1, r, r_in) + rec_CX_count_assym_punc(r-1, m-1, r_in, m_in, state_prep=state_prep) + rec_CX_count_assym(r-1, m-1, r_in, m_in)