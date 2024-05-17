### Recursive circuits with new construction

import tequila as tq
from permutations import *
from qrm_circuits import get_qubit_partition
from qrm_matrices import Grm, GeneratorQuotient
import numpy as np

def RowIndexList(G):
    return {tuple(row): i for i, row in enumerate(G)}

def RecursiveBasisQRM_m(m, ql = [], partitions=[]):
    if ql == []:
        ql = list(range(2**m))
    if m == 0:
        return tq.QCircuit(), Permutation({ql[0]: ql[0]}), RowIndexList(Grm(0, 0))
    
    ql1, ql2, par1, par2 = get_qubit_partition(m, ql, partitions=partitions)
    
    U1, P1, M1 = RecursiveBasisQRM_m(m-1, ql1, par1)
    U2, P2, M2 = RecursiveBasisQRM_m(m-1, ql2, par2)
    circuit = tq.QCircuit()

    G = Grm(m, m)
    M = RowIndexList(G)
    G1 = Grm(m-1, m-1)

    P = {}
    for row in G1:
        u = tuple(row)
        uu = tuple(np.concatenate([u, u]))
        z = np.zeros(2**(m-1), dtype=int)
        zu = tuple(np.concatenate([z, u]))

        P[ql[M[uu]]] = ql1[M1[u]]
        P[ql[M[zu]]] = ql2[M2[u]]

        circuit += tq.gates.CNOT(control=ql1[M1[u]], target=ql2[M2[u]])
    
    P = Permutation(P)
    P.fill(qubit_list=ql)
    PP = P1 + P2

    circuit = PP.permute_circuit(circuit)
    circuit = circuit + U1 + U2
    P_final = PP * P
    
    return circuit, P_final, M

def RecursiveBasisQRM(r, m, ql = [], partitions=[]):
    """
    Recursive encoder for computational basis states of RM(r, m)
    """
    if ql == []:
        ql = list(range(2**m))
    if r == m:
        return RecursiveBasisQRM_m(m, ql = ql, partitions=partitions) 
    if m == 0:
        return tq.QCircuit(), Permutation({ql[0]: ql[0]}), RowIndexList(Grm(0, 0))
    
    ql1, ql2, par1, par2 = get_qubit_partition(m, ql, partitions=partitions)
    
    U1, P1, M1 = RecursiveBasisQRM(r, m-1, ql1, par1)
    U2, P2, M2 = RecursiveBasisQRM(r, m-1, ql2, par2)
    circuit = tq.QCircuit()

    G = Grm(r, m)
    M = RowIndexList(G)
    G1 = Grm(r, m-1)
    G2 = Grm(r-1, m-1)

    P = {}
    for row in G1:
        u = tuple(row)
        uu = tuple(np.concatenate([u, u]))
        P[ql[M[uu]]] = ql1[M1[u]]
        circuit += tq.gates.CNOT(control=ql1[M1[u]], target=ql2[M2[u]])
    
    for row in G2:
        v = tuple(row)
        z = np.zeros(2**(m-1), dtype=int)
        zv = tuple(np.concatenate([z, v]))
        P[ql[M[zv]]] = ql2[M2[v]]
    
    P = Permutation(P)
    P.fill(qubit_list=ql)
    PP = P1 + P2

    circuit = PP.permute_circuit(circuit)
    circuit = circuit + U1 + U2
    P_final = PP * P

    return circuit, P_final, M

def RecursiveQRM(r, m, ql = [], partitions = []):
    if ql == []:
        ql = list(range(2**m))
    if r > m:
        raise ValueError("Incorrect parameters r>m!")
    if r == m:
        return RecursiveBasisQRM(r, m, partitions=partitions, ql=ql)
    
    circuit = tq.QCircuit()
    ql1, ql2, par1, par2 = get_qubit_partition(m, ql, partitions)

    U1, P1, M1 = RecursiveQRM(r, m-1, ql1, par1)
    U2, P2, M2 = RecursiveQRM(r, m-1, ql2, par2)

    #form generator matrix and M
    G1 = GeneratorQuotient(r, m-r-1, m)
    G2 = GeneratorQuotient(m-r-1, m-r-2, m-1)
    G2G2 = [np.concatenate([row, row]) for row in G2]
    G = G1 + G2G2
    M = RowIndexList(G)

    #form permutation and circuit
    G3 = GeneratorQuotient(r, m-r-2, m-1)
    G4 = GeneratorQuotient(r-1, m-r-2, m-1)
    
    P = {}
    circuit_H = tq.QCircuit()
    for row in G2G2:
        uu = tuple(row)
        circuit_H += tq.gates.H(ql[M[uu]])
    
    for row in G3:
        u = tuple(row)
        uu = tuple(np.concatenate([u, u]))
        
        P[ql[M[uu]]] = ql1[M1[u]]
        circuit += tq.gates.CNOT(control=ql1[M1[u]], target=ql2[M2[u]])

    for row in G4:
        z = np.zeros(2**(m-1), dtype=int)
        v = tuple(row)
        zv = tuple(np.concatenate([z, v]))

        P[ql[M[zv]]] = ql2[M2[v]]
    
    P = Permutation(P)
    P.fill(qubit_list=ql)
    PP = P1 + P2

    circuit_H = P.permute_circuit(circuit_H)
    circuit = circuit_H + circuit
    circuit = PP.permute_circuit(circuit)
    circuit = circuit + U1 + U2
    P_final = PP * P
    
    return circuit, P_final, M