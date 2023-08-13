#Utilities for QRM gate counts and circuit constructions
#Praveen Jayakumar, July 2023

import math
import numpy as np
import copy
import tequila
from qiskit import QuantumCircuit
from qiskit.tools.visualization import circuit_drawer

def binom_sum(m,start,end):
    #includes start and end
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

def get_eval_set(row, reversed = False):
    '''
    Returns the evaluation set for a evaluation vector  
    [0 0 0 1 0 0 0 1] = Eval^3(x_1x_2) returns [1, 2]  
    [0 0 0 0 0 0 1 1] = Eval^3(x_0x_1) returns [0, 1]  

    x subscript starting with x_0
    '''
    m = int(np.log2(len(row)))
    i = leading_bit_index(row)
    return get_int_set(i, m, reversed=reversed)

def get_int_set(i, m, reversed = False):
    '''
    Returns set representation of i, ie the bit positions non-zero of bin(i)
    reverse: reversed bit positions

    i = 6, m = 5 bin(6) = 00110
    reversed = True returns [1, 2]
    reversed = False return [2, 3]
    '''

    assert i < 1<<m, print('Invalid parameter i: {}'.format(i))

    s = []
    n = 0
    while i > 0:
        if i%2 ==1:
            if reversed:
                s.append(n)
            else:
                s.append(m - n - 1)
        i = i//2
        n+=1
    if not reversed:
        s.reverse()
    return s

def get_list_int(l):
    return

def add_to_set(s, to_add = []):
    '''
    Adds elements of to_add to s and extends m to m+1

    s = [0, 1, 3] to_add = [1] returns [0, 1, 2, 4]
    '''
    if to_add == []:
        return s
    
    l = get_set_list(s)
    for ind in to_add:
        l.insert(ind, 1)
    return get_list_set(l)

def get_list_set(l):
    s = []
    for ind, i in enumerate(l):
        if i == 1:
            s.append(ind)
    return s
        
def add_to_list_elements(l = [], i = 0):
    '''
    shift list elements by i
    '''
    return [a + i for a in l]

def add_to_dict_elements(dict_, i = 0):
    '''
    Add to the dictionaru elements

    '''
    dict_new = {}
    for k, v in zip(dict_.keys(), dict_.values()):
        dict_new[k] = v + i
    return dict_new

def add_dict(dict_1, dict_2):
    '''
    Combine two python dictionaries

    Defualts to combining values into list if same key, 
    '''
    dict_new = {}
    for k in dict_1.keys():
        dict_new[k] = dict_1[k]
    
    for k, v in zip(dict_2.keys(), dict_2.values()):
        if k in dict_1.keys():
            dict_new[k] = [dict_1[k], v]
        else:
            dict_new[k] = v
    return dict_new

def get_dict_values(dict_, keys = []):
    '''
    Return values given by keys from dict_
    Skips if key value not present
    '''
    values = []
    for k in keys:
        values.append(dict_[k])
    return values

def add_ind_to_key_ind(dict_, m, to_add = [0], reversed = True):
    '''
    Add the 
    '''
    dict_new = {}
    for k, v in zip(dict_.keys(), dict_.values()):
        k_new = get_set_int(add_to_set(get_int_set(k, m = m, reversed=reversed), to_add))
        dict_new[k_new] = v
    return dict_new

def add_key_value_dict(dict_, to_add_key = 0, to_add_value = 0):
    '''
    Adds to key and value of dictionary
    '''
    return {ind + to_add_key: pos + to_add_value for ind, pos in zip(dict_.keys(), dict_.values())}

def get_set_int(l):
    '''
    Returns integer representation of set of monomial subscripts

    int representation of set [0, 2] is 5
    int representation of [] (Eval(1)) is 0
    '''
    return int(np.sum([1<<i for i in l]))

def get_set_list(s, m = None):
    '''
    Returns list indicating 
    '''
    int_ = get_set_int(s)
    bin_i = bin(int_)[2:]
    l = [int(b) for b in bin_i]
    l.reverse()
    return l

def min_set(in_ind, r, as_int = False):
    '''
    Returns minimum superset of in_ind with cardinality r
    as_int if True returns the int representation of the superset
    '''
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
        return [get_set_int(se) for se in s]
    return s

def get_qiskit_circuit(circuit):
    qasmstr = tequila.export_open_qasm(circuit)
    qiskit_cir = QuantumCircuit.from_qasm_str(qasmstr)
    return qiskit_cir

def draw_tequila_circuit(circuit, justify = 'left'):
    qc = get_qiskit_circuit(circuit)
    return circuit_drawer(qc, fold=-1, justify = justify)

def check_if_same_circuit(circuit_1, circuit_2, tol = 1e-6):
    '''
    checks if two tequila circuits are the same by simulating
    #todo: do this with stim for more qubits and faster simulation.
    '''
    wf1 = tequila.simulate(circuit_1)
    wf2 = tequila.simulate(circuit_2)
    wf3 = wf1 - wf2
    diff = sum(np.abs(list(wf3.values())))
    return diff < tol

def puncture_row(row, remove_ind = [0]):
    '''
    Puncture row by removing indices listed in remove_ind
    '''
    new_row = []
    for i, r in enumerate(row):
        if i not in remove_ind:
            new_row.append(r)
    return new_row

def puncture_matrix(M, remove_ind = [0]):
    '''
    Puncture matrix by removing columns specified by remove_ind
    '''
    return [puncture_row(row, remove_ind=remove_ind) for row in M]

def connectivity(circuit, n_qubit):
    '''
    Returns (list((control connectivity, target connectivity)), Ed)
    Assuming input circuit with only CNOT gates

    ''' 
    if len(circuit.gates) == 0:
        return [{'x': set(), 'z': set()} for _ in range(n_qubit)], 0
    
    #recursively call
    gate = circuit.gates[0]
    rec_circuit = tequila.QCircuit(gates = circuit.gates[1:])
    conn_res, Ed = connectivity(rec_circuit, n_qubit)

    #add gate
    assert gate.name.lower() == 'x', 'Gate not CNOT, error.'
    assert len(gate.control) == 1, 'Gate has incorrect controls'
    assert len(gate.target) == 1, 'Gate has incorrect targets'
    
    control = gate.control[0]
    target = gate.target[0]

    #combine
    conn_res[control]['x'] = conn_res[control]['x'].union(conn_res[target]['x'])
    conn_res[control]['x'].add(target)
    conn_res[target]['z'] = conn_res[target]['z'].union(conn_res[control]['z'])
    conn_res[target]['z'].add(control)

    Ed = np.average([len(conn_res[i]['x'].union(conn_res[i]['z'])) for i in range(n_qubit)])
    return conn_res, Ed