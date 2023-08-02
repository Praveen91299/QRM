#circuits for QRM codes
#Praveen Jayakumar, July 2023

import tequila
import stim
from qrm_utils import *
from qrm_matrices import *
from stim_utils import stim_CNOT_list, stim_H_list, tequila_to_stim

def canonical_CSS(G, n_qubits, qubit_list = None, only_cnots = False):
    '''
    qubit_list is the qubit map. 
    Eg: qubit_list = [2, 3, 5, 6] - the circuits will be over these 4 qubits only
    '''
    Gperp, G1q = G
    if qubit_list != None:
        if len(qubit_list) != n_qubits:
            print('Insufficient qubits provided.')
            return
    else:
        qubit_list = list(range(n_qubits))
    circuit = tequila.QCircuit() #qubit indices?
    #returns circuit and message qubit indexes
    cnots = []
    ent_qubits = []
    msg_qubit = []
    for row in Gperp:
        i = leading_bit_index(row)
        ent_qubits.append(qubit_list[i])
        if not only_cnots:
            circuit += tequila.gates.H(target = qubit_list[i])
        targets = [qubit_list[j] for j in get_indexes(row)][1:]
        cnots.append([i, tequila.gates.CX(control = qubit_list[i], target = targets)])
    
    for row in G1q:
        i = leading_bit_index(row)
        msg_qubit.append(qubit_list[i])
        targets = [qubit_list[j] for j in get_indexes(row)][1:]
        cnots.append([i, tequila.gates.CX(control = qubit_list[i], target = targets)])
    
    sorted_cnots = sorted(cnots, key=lambda a: a[0], reverse=True)
    for c in sorted_cnots:
        circuit += c[1]
    return circuit, (msg_qubit, ent_qubits)

def QRM_std_circuit(r, m, qubit_list = None, transform_rows = True, only_cnots = False):
    '''
    set transform_rows = False for Naive encoder.
    '''
    def R_G(G):
        if len(G) == 0:
            return G
        R = get_R(G)
        return np.einsum('ap, pb', R, G) % 2
    params = (r, m)
    Gs = get_QRM_generator(params, params)
    Gperpnew = R_G(Gs[0])
    G1qnew = R_G(Gs[1])
    if transform_rows:
        Gsnew = (Gperpnew, G1qnew)
    else:
        Gsnew = Gs
    circuit, qubit_info = canonical_CSS(Gsnew, n_qubits = 2**m, only_cnots = only_cnots, qubit_list=qubit_list)
    return circuit, qubit_info

def QRM_punc_std_circuit(r, m, qubit_list = None, transform_rows = True, only_cnots = False):
    '''
    set transform_rows = False for Naive encoder.
    '''
    def R_Gs(Gl):
        
        if len(G) == 0:
            return G
        R = get_R_punc(G)
        return np.einsum('ap, pb', R, G) % 2
    params = (r, m)
    Gs = get_QRM_punc_generator(params, params)
    Gs_transformed = R_Gs(Gs)
    if transform_rows:
        Gsnew = Gs_transformed
    else:
        Gsnew = Gs
    circuit, qubit_info = canonical_CSS(Gsnew, n_qubits = 2**m, only_cnots = only_cnots, qubit_list=qubit_list)
    return circuit, qubit_info

class QRMcircuit:
    '''
    Circuit object for recursive QRM encoders, complete when required.
    '''
    def __init__(self):
        self.circuit = tequila.QCircuit()
    
    def CX(self, control=[], target=[]):
        self.circuit += tequila.gates.CX(control=control, target=target)
    
    def H(self, target=[]):
        self.circuit += tequila.gates.H(target=target)
    
    def get_circuit(self, circ_type ='tequila'):
        if circ_type == 'tequila':
            return self.circuit
        elif circ_type == 'stim':
            return tequila_to_stim(self.circuit)
    
    @property
    def depth(self):
        return stim_to_tequila(self.circuit).depth
    
    @property
    def gate_count(self):
        return len(stim_to_tequila(self.circuit).gates)
    
    def __add__(self, other):
        result_circuit = self.circuit + other.circuit
        return QRMcircuit(result_circuit)
    
    def __repr__(self) -> str:
        return self.circuit.__repr__()

def get_qubit_partition(m, qubit_list = None, partitions = []):
    if qubit_list == None:
        qubit_list = list(range(2**m))
    
    if partitions != []:
        ql_1, ql_2 = apply_qubit_partition(partitions[0], m, qubit_list)
        if len(partitions) > 1:
            p1 = partitions[1]
            if len(partitions) > 2:
                p2 = partitions[2]
            else:
                p2 = []
        else:
            p1, p2 = [], []
    else:
        ql_1, ql_2 = qubit_list[:2**(m-1)], qubit_list[2**(m-1):]
        p1, p2 = [], []
    return ql_1, ql_2, p1, p2

def get_punc_qubit_partition(m, qubit_list=None, partitions=[], punc_bit_list=[0]):
    if qubit_list == None:
        qubit_list = list(range((2**m)-len(punc_bit_list)))
    
    if partitions != []:
        ql_1, ql_2 = apply_punc_qubit_partition(partitions[0], m, qubit_list=qubit_list, punc_bit_list=punc_bit_list)
        if len(partitions) > 1:
            p1 = partitions[1]
            if len(partitions) > 2:
                p2 = partitions[2]
            else:
                p2 = []
        else:
            p1, p2 = [], []
    else:
        ql_1, ql_2 = apply_punc_qubit_partition(1, m, qubit_list=qubit_list, punc_bit_list=punc_bit_list)
        p1, p2 = [], []
    return ql_1, ql_2, p1, p2

def add_entanglers(r1, r2, m, ql_1 = None, ql_2 = None, only_cnots = True):
    '''
    Add entanglers from ql_1 to ql_2 depending on leading bits of Gm(r1, r2, m). Set only_cnots = False for Hadamard gates on the first set
    ql_1, ql_2 of length 2**m
    '''
    U = tequila.QCircuit()
    
    if ql_1 == None:
        ql_1 = list(range(2**m))
    if ql_2 == None:
        ql_2 = list(range(2**m, 2**(m+1)))
    
    Gm = get_QRM_generators_r1r2(r1, r2, m)
    lbs = [leading_bit_index(row) for row in Gm]
    qm1 = [ql_1[i] for i in lbs]
    qm2 = [ql_2[i] for i in lbs]
    if not only_cnots:
        U += tequila.gates.H(target=qm1)
    for a, b in zip(qm1, qm2):
        U += tequila.gates.CX(control=a, target=b)
    
    return U

def add_punc_entanglers(r1, r2, m, ql_1 = None, ql_2 = None, only_cnots = True):
    '''
    Add entanglers for punctured codes
    Currently defaulted to first bit
    
    '''
    U = tequila.QCircuit()

    if ql_1 == None:
        ql_1 = list(range((2**m) - 1))
    if ql_2 == None:
        ql_2 = list(range(2**m -1, 2**(m+1) - 1))
    
    if r1 <= 0:
        print('Not handled right now.')
        return
    Gm = get_QRM_generators_r1r2(r1, r2, m)
    Gp = puncture_matrix(Gm, remove_ind=[0])
    lb1 = [leading_bit_index(row) for row in Gp]
    lb2 = [leading_bit_index(row) for row in Gm]
    qm1 = [ql_1[i] for i in lb1]
    qm2 = [ql_2[i] for i in lb2]
    if not only_cnots:
        U += tequila.gates.H(target=qm1)
    for a, b in zip(qm1, qm2):
        U += tequila.gates.CX(control=a, target=b)
    
    return U

def add_hadamards(r1, r2, m, qubit_list = None):
    if qubit_list == None:
        qubit_list = list(range(2**m))

    Gm = get_QRM_generators_r1r2(r1, r2, m)
    lbs = [leading_bit_index(row) for row in Gm]
    qm = [qubit_list[i] for i in lbs]
    return tequila.gates.H(target=qm)


def QRM_rec_classical_circuit(r, m, partitions = [], qubit_list = None):
    U = tequila.QCircuit()

    if m == 0 or r == -1:
        return U, None

    if r > m or m < 0 or r < -1:
        print('Invalid parameters r, m : {}, {}'.format(r, m))
        return U, None
    
    
    ql_1, ql_2, p1, p2 = get_qubit_partition(m, qubit_list=qubit_list, partitions=partitions)

    if r == m:
        U += add_entanglers(0, r-1, m-1, ql_1=ql_1, ql_2=ql_2, only_cnots=True)

        U1, info_1 = QRM_rec_circuit(r-1, m-1, partitions=p1, qubit_list=ql_1)
        U2, info_2 = QRM_rec_circuit(r-1, m-1, partitions=p2, qubit_list=ql_2)
        U = U + U1 + U2
        return U, None
        
    U1, info_1 = QRM_rec_classical_circuit(r, m-1, partitions=p1, qubit_list=ql_1)
    U2, info_2 = QRM_rec_classical_circuit(r, m-1, partitions=p2, qubit_list=ql_2)
    
    U += add_entanglers(0, r, m-1, ql_1=ql_1, ql_2=ql_2, only_cnots=True)

    U += U1
    U += U2
    return U, None    

def QRM_rec_circuit(r, m, partitions = [], qubit_list = None, only_cnots = False):

    U = tequila.QCircuit()
    msg_q = []
    ent_q = []
    if m == 0:
        return U, (qubit_list, [])
    
    if r > m or r < m-r-1:
        print('Invalid parameters r, m : {}, {}'.format(r, m))
        return U, None
    
    if r == m:
        return QRM_rec_classical_circuit(r, m, partitions=partitions, qubit_list=qubit_list)
    
    ql_1, ql_2, p1, p2 = get_qubit_partition(m, qubit_list=qubit_list, partitions=partitions)

    if r > m-r-1:
        #add message ents
        U += add_entanglers(m-r, r, m-1, ql_1=ql_1, ql_2=ql_2, only_cnots=True)
    
    #add entanglers
    U += add_entanglers(m-r-1, m-r-1, m-1, ql_1=ql_1, ql_2=ql_2, only_cnots=only_cnots)

    U1, info_1 = QRM_rec_circuit(r, m-1, partitions=p1, qubit_list=ql_1, only_cnots=only_cnots)
    U2, info_2 = QRM_rec_circuit(r, m-1, partitions=p2, qubit_list=ql_2, only_cnots=only_cnots)

    U += U1
    U += U2
    
    return U, None

def QRM_rec_assym_circuit(r, m, r_in, m_in, partitions = [], qubit_list = None, only_cnots = False):
    U = tequila.QCircuit()
    
    if m == 0:
        return U, (qubit_list, [])
    
    if r > m or r < -1:
        print('Invalid parameters r, m : {}, {}'.format(r, m))
        return U, None
    
    if 2*r_in + 1 <= m_in:
        if r == -1:
            Uc, info = QRM_rec_classical_circuit(r_in, m, partitions=partitions, qubit_list=qubit_list)
            return Uc, None
    if 2*r_in + 1 > m_in:
        if m == r_in:
            #add Hadamards
            if not only_cnots:
                U += add_hadamards(0, 2*r_in - m_in, m, qubit_list=qubit_list)
            #classical circuit
            Uc, i = QRM_rec_classical_circuit(r_in, r_in, partitions=partitions, qubit_list=qubit_list)
            return U + Uc, None
    
    ql_1, ql_2, p1, p2 = get_qubit_partition(m, qubit_list=qubit_list, partitions=partitions)
    
    if r != r_in:
        U += add_entanglers(r+1, r_in, m-1, ql_1=ql_1, ql_2=ql_2, only_cnots = True)
        U += add_entanglers(r, r, m-1, ql_1=ql_1, ql_2=ql_2, only_cnots=only_cnots)
    else:
        #initial entanglers
        U += add_entanglers(r, r_in, m-1, ql_1=ql_1, ql_2=ql_2, only_cnots = only_cnots)
    
    U1, info_1 = QRM_rec_assym_circuit(r-1, m-1, r_in, m_in, partitions=p1, qubit_list=ql_1, only_cnots=only_cnots)
    U2, info_2 = QRM_rec_assym_circuit(r-1, m-1, r_in, m_in, partitions=p2, qubit_list=ql_2, only_cnots=only_cnots)

    U += U1
    U += U2

    return U, None

def QRM_rec_punc_circuit(r, m, partitions = [], qubit_list = None, only_cnots = False, state_prep = False):
    '''
    Punctured QRM encoder, currently defaulted to dropping first qubit q[0]
    
    '''
    U = tequila.QCircuit()
    punc_bit_list = [0]

    if r >= m or r < m-r-1:
        print('Invalid parameters r, m : {}, {}'.format(r, m))
        return U, None
    
    ql_1, ql_2, p1, p2 = get_punc_qubit_partition(m, qubit_list=qubit_list, partitions=partitions, punc_bit_list=punc_bit_list)

    if m == r + 1:
        #add U^*(r, r+1)
        if r == 0:
            return U, None
        Up, info = QRM_rec_punc_circuit(r-1, m-1, partitions=p1, qubit_list=ql_1, only_cnots=True, state_prep=state_prep)
        U += Up

        U += tequila.gates.CX(control=ql_1[-1], target=ql_2[-1])
        Uc, info = QRM_rec_classical_circuit(r, r, partitions=p2, qubit_list=ql_2)
        U += Uc
        return U, None
    
    if r > m-r-1:
        #add message ents
        U += add_punc_entanglers(m-r, r, m-1, ql_1=ql_1, ql_2=ql_2, only_cnots=True)
    
    #add entanglers
    U += add_punc_entanglers(m-r-1, m-r-1, m-1, ql_1=ql_1, ql_2=ql_2, only_cnots=only_cnots)
    
    U1, info_1 = QRM_rec_punc_circuit(r, m-1, partitions=p1, qubit_list=ql_1, only_cnots=only_cnots)
    U2, info_2 = QRM_rec_circuit(r, m-1, partitions=p2, qubit_list=ql_2, only_cnots=only_cnots)

    U += U1
    U += U2

    return U, None

def QRM_rec_assym_punc_circuit(r, m, r_in, m_in, partitions = [], qubit_list = None, only_cnots = False):
    
    return