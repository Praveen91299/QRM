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
    Gate counts for recursive U_rr
    """
    if r ==1:
        return 1 
    g = 2**(r-1)
    return g + 2*Urr_CX_count(r-1)

def rec_CX_count(r, m):
    if r>m:
        print('Invalid parameters r = {} > m = {}'.format(r, m))
        return
    if r==m:
        return Urr_CX_count(r)
    if m-r-1 == r:
        return math.comb(m-1, m-r-1) + 2*rec_CX_count(r, m-1)
    else:
        return binom_sum(m-1, m-r, r) + math.comb(m-1, m-r-1) + 2*rec_CX_count(r, m-1)

def Urm_CX_count(r, m):
    if r == 0:
        return 2**m - 1
    if r == m:
        return Urr_CX_count(m)
    return binom_sum(m-1, 0, r) + 2*Urm_CX_count(r, m-1)  # (u, u) + (0, v) -> (u, 0) + (0, u+v)

def rec_CX_count_assym(r, m, r_in, m_in):
    #gate counts for assymmetric QRM code
    if 2*r_in + 1 > m_in:
        if m == r_in:
            return Urr_CX_count(m)
    if 2*r_in + 1 <=m_in:
        if r == -1:
            return Urm_CX_count(r_in, m)
    return binom_sum(m-1, r, r_in) + 2*rec_CX_count_assym(r-1, m-1, r_in, m_in)

def rec_punc(r, m):
    return 

def rec_punc_zero(r, m, rin):
    #incomplete!
    return binom_sum(m-1, r, rin) + 1 + rec_punc_zero(r-1, m-1, rin)

def Urr_CX_count_punc_no1(r):
    """
    Gate counts for recursive U_rr
    """
    if r ==1:
        return 0
    g = 2**(r-1) - 1
    return g + Urr_CX_count_punc_no1(r-1) + Urr_CX_count(r-1)

def rec_CX_count_assym_punc(r, m, rin, state_prep = False):
    #gate counts for assymmetric QRM code
    if 2*r + 1 > m:
        return
    if r ==-1:
        if state_prep:
            return Urr_CX_count_punc_no1(m)
        return Urr_CX_count(m)
    if state_prep:
        return binom_sum(m-1, r, rin) + rec_CX_count_assym_punc(r-1, m-1, rin, state_prep=state_prep) + rec_CX_count_assym(r-1, m-1, rin)
    return binom_sum(m-1, r, rin) + 1 + rec_CX_count_assym_punc(r-1, m-1, rin, state_prep=state_prep) + rec_CX_count_assym(r-1, m-1, rin)