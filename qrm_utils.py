#code for QRM gate counts and circuit constructions
#Praveen Jayakumar, July 2023

import math
import numpy as np
import copy
import tequila
from qiskit import QuantumCircuit

def binom_sum(m,start,end):
    if start > end:
        print('Error, start > end')
        return
    if start > m or end > m:
        print('Print comb start or end larger than m')
        return
    return np.sum([math.comb(m, i) for i in range(start, end + 1)])

def reorder_wt(G, reverse = True):
    #orders rows with weight
    return np.array(sorted(G, key=lambda x: np.sum(x), reverse=reverse))

def filter_wt(G, weights):
    Gnew = []
    for row in G:
        if sum(row) in weights:
            Gnew.append(row)
    return Gnew

def get_indexes(row):
    l = []
    for i, r in enumerate(row):
        if r == 1:
            l.append(i)
    return l

def leading_bit_index(row):
    N = len(row)
    for i in range(N):
        if row[i] == 1:
            return i
    return N-1

def get_eval_set(row):
    m = int(np.log2(len(row)))
    i = leading_bit_index(row)
    bin_i = bin(i)[2:]
    s = []
    n = 0
    while i > 0:
        if i%2 ==1:
            s.append(n)
        i = i//2
        n+=1
    return s

def get_int(l):
    '''
    int representation of list [1, 0, 1, 0] is 5
    '''
    return np.sum([1<<i for i in l])

def min_set(in_ind, r, as_int = False):
    ind = copy.copy(in_ind)
    s = [copy.copy(ind)]
    for i in range(len(ind) + r):
        if len(ind) >= r:
            break
        if i not in ind:
            new = [se + [i] for se in s]
            s = new + s
            ind.append(i)
    if as_int:
        return [get_int(se) for se in s]
    return s

def get_qiskit_circuit(circuit):
    qasmstr = tequila.export_open_qasm(circuit)
    qiskit_cir = QuantumCircuit.from_qasm_str(qasmstr)
    return qiskit_cir