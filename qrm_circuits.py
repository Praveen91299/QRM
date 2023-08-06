#circuits for QRM codes
#Praveen Jayakumar, July 2023

import tequila
import stim
from qrm_utils import *
from qrm_matrices import *
from stim_utils import stim_CNOT_list, stim_H_list, tequila_to_stim

### normal constructions

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
    set transform_rows = False for Naive encoder. #INCOMPLETE RIGHT NOW
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

### recursive constructions

class QRMcircuit:
    '''
    Circuit object for recursive QRM encoders, complete when required!
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
        return self.circuit.depth
    
    @property
    def gate_count(self):
        return len(self.circuit.gates)
    
    def __add__(self, other):
        result_circuit = self.circuit + other.circuit
        return QRMcircuit(result_circuit)
    
    def __repr__(self) -> str:
        return self.circuit.__repr__()

def get_qubit_partition(m, qubit_list = None, partitions = []):
    if qubit_list == None:
        qubit_list = list(range(2**m))
    
    if partitions != []:
        ql_1, ql_2 = apply_qubit_partition(partitions[0], m, qubit_list=qubit_list)
        if len(partitions) > 1:
            p1 = partitions[1]
            if len(partitions) > 2:
                p2 = partitions[2]
            else:
                p2 = []
        else:
            p1, p2 = [], []
    else:
        ql_1, ql_2 = apply_qubit_partition(0, m, qubit_list=qubit_list)
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
        ql_1, ql_2 = apply_punc_qubit_partition(0, m, qubit_list=qubit_list, punc_bit_list=punc_bit_list)
        p1, p2 = [], []
    return ql_1, ql_2, p1, p2

def add_entanglers(r1, r2, m, ql_1 = None, ql_2 = None, pos_dict1 = None, pos_dict2 = None, only_cnots = True):
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
    if pos_dict1 is not None:
        lb1 = get_dict_values(pos_dict1, keys = lbs)
    if pos_dict2 is not None:
        lb2 = get_dict_values(pos_dict2, keys = lbs)
    
    qm1 = [ql_1[i] for i in lb1]
    qm2 = [ql_2[i] for i in lb2]
    if not only_cnots:
        U += tequila.gates.H(target=qm1)
    for a, b in zip(qm1, qm2):
        U += tequila.gates.CX(control=a, target=b)
    
    return U

def add_punc_entanglers(r1, r2, m, ql_1 = None, ql_2 = None, pos_dict1 = None, pos_dict2 = None, only_cnots = True, puncture_Gp = False):
    '''
    Add entanglers for punctured codes
    Currently defaulted to first bit

    ql_1/ql_2: qubit positions of the two halves in actual circuit
    pos_dict1/pos_dict2: row reordering mapping {row/leading index: actual row index in permuted G}
    puncture_Gp: to determine control indexes from punctured/unpunctured generator. Set False if index position already punctured.
    
    '''
    U = tequila.QCircuit()

    if ql_1 == None:
        ql_1 = list(range((2**m) - 1))
    if ql_2 == None:
        ql_2 = list(range(2**m -1, 2**(m+1) - 1))
    
    assert r2 >=r1, 'Invalid parameters r2 < r1.'

    if r1 < 0:
        print('Invalid parameter r1: {}'.format(r1))
        return
    
    Gm = get_QRM_generators_r1r2(r1, r2, m)
    if puncture_Gp:
        Gp = puncture_matrix(Gm, remove_ind=[0])
    else:
        Gp = Gm
    
    lb1 = [leading_bit_index(row) for row in Gp]
    lb2 = [leading_bit_index(row) for row in Gm]
    if pos_dict1 is not None:
        lb1 = get_dict_values(pos_dict1, keys = lb1)
    if pos_dict2 is not None:
        lb2 = get_dict_values(pos_dict2, keys = lb2)
    
    qm1 = [ql_1[i] for i in lb1]
    qm2 = [ql_2[i] for i in lb2]
    if not only_cnots:
        U += tequila.gates.H(target=qm1)
    for a, b in zip(qm1, qm2):
        U += tequila.gates.CX(control=a, target=b)
    
    return U

def add_hadamards(r1, r2, m, qubit_list = None, pos_dict = None):
    if qubit_list == None:
        qubit_list = list(range(2**m))

    Gm = get_QRM_generators_r1r2(r1, r2, m)
    lbs = [leading_bit_index(row) for row in Gm]
    if pos_dict is not None:
        lbs = [pos_dict[i] for i in lbs]
    
    qm = [qubit_list[i] for i in lbs]
    return tequila.gates.H(target=qm)


def QRM_rec_classical_circuit(r, m, partitions = [], qubit_list = None):
    U = tequila.QCircuit()

    if m == 0 or r == -1:
        return U, {0 : 0}

    if r > m or m < 0 or r < -1:
        print('Invalid parameters r, m : {}, {}'.format(r, m))
        return U, None
    
    
    ql_1, ql_2, p1, p2 = get_qubit_partition(m, qubit_list=qubit_list, partitions=partitions)

    if r == m:
        U1, position_dict1 = QRM_rec_classical_circuit(r-1, m-1, partitions=p1, qubit_list=ql_1)
        U2, position_dict2 = QRM_rec_classical_circuit(r-1, m-1, partitions=p2, qubit_list=ql_2)

        U += add_entanglers(0, r-1, m-1, ql_1=ql_1, ql_2=ql_2, pos_dict1=position_dict1, pos_dict2=position_dict2, only_cnots=True)
        U = U + U1 + U2

        position_dict2 = add_key_value_dict(position_dict2, 2**(m-1), len(position_dict1.keys()))
        position_dict_new = add_dict(position_dict1, position_dict2)
        return U, position_dict_new
        
    U1, position_dict1 = QRM_rec_classical_circuit(r, m-1, partitions=p1, qubit_list=ql_1)
    U2, position_dict2 = QRM_rec_classical_circuit(r, m-1, partitions=p2, qubit_list=ql_2)
    
    U += add_entanglers(0, r, m-1, ql_1=ql_1, ql_2=ql_2, pos_dict1=position_dict1, pos_dict2=position_dict2, only_cnots=True)
    U += U1
    U += U2

    position_dict2 = add_key_value_dict(position_dict2, 2**(m-1), len(position_dict1.keys()))
    position_dict_new = add_dict(position_dict1, position_dict2)

    return U, position_dict_new 

def QRM_rec_circuit(r, m, partitions = [], qubit_list = None, only_cnots = False, classical=False):
    if classical:
        return QRM_rec_classical_circuit(r, m, partitions=partitions, qubit_list=qubit_list)

    U = tequila.QCircuit()
    if m == 0:
        return U, {0 : 0}
    
    if r > m or r < m-r-1:
        print('Invalid parameters r, m : {}, {}'.format(r, m))
        return U, None
    
    if r == m:
        return QRM_rec_classical_circuit(r, m, partitions=partitions, qubit_list=qubit_list)
    
    ql_1, ql_2, p1, p2 = get_qubit_partition(m, qubit_list=qubit_list, partitions=partitions)

    U1, position_dict1 = QRM_rec_circuit(r, m-1, partitions=p1, qubit_list=ql_1, only_cnots=only_cnots)
    U2, position_dict2 = QRM_rec_circuit(r, m-1, partitions=p2, qubit_list=ql_2, only_cnots=only_cnots)

    if r > m-r-1:
        #add message ents
        U += add_entanglers(m-r, r, m-1, ql_1=ql_1, ql_2=ql_2, pos_dict1=position_dict1, pos_dict2=position_dict2, only_cnots=True)
    
    #add entanglers
    U += add_entanglers(m-r-1, m-r-1, m-1, ql_1=ql_1, ql_2=ql_2, pos_dict1=position_dict1, pos_dict2=position_dict2, only_cnots=only_cnots)

    U += U1
    U += U2
    
    position_dict2 = add_key_value_dict(position_dict2, to_add_key=2**(m-1), to_add_value=len(position_dict1.keys()))
    position_dict_new = add_dict(position_dict1, position_dict2)

    return U, position_dict_new

def QRM_rec_assym_circuit(r, m, r_in, m_in, partitions = [], qubit_list = None, only_cnots = False):
    U = tequila.QCircuit()
    
    if m == 0:
        return U, {0: 0}
    
    if r > m or r < -1:
        print('Invalid parameters r, m : {}, {}'.format(r, m))
        return U, None
    
    if 2*r_in + 1 <= m_in:
        if r == -1:
            return QRM_rec_classical_circuit(r_in, m, partitions=partitions, qubit_list=qubit_list)  
    if 2*r_in + 1 > m_in:
        if m == r_in:
            if not only_cnots:
                U += add_hadamards(0, 2*r_in - m_in, m, qubit_list=qubit_list)
            Uc, position_dict = QRM_rec_classical_circuit(r_in, r_in, partitions=partitions, qubit_list=qubit_list)
            return U + Uc, position_dict
    
    ql_1, ql_2, p1, p2 = get_qubit_partition(m, qubit_list=qubit_list, partitions=partitions)
    
    U1, position_dict1 = QRM_rec_assym_circuit(r-1, m-1, r_in, m_in, partitions=p1, qubit_list=ql_1, only_cnots=only_cnots)
    U2, position_dict2 = QRM_rec_assym_circuit(r-1, m-1, r_in, m_in, partitions=p2, qubit_list=ql_2, only_cnots=only_cnots)

    if r != r_in:
        U += add_entanglers(r+1, r_in, m-1, ql_1=ql_1, ql_2=ql_2, pos_dict1=position_dict1, pos_dict2=position_dict2, only_cnots = True)
        U += add_entanglers(r, r, m-1, ql_1=ql_1, ql_2=ql_2, pos_dict1=position_dict1, pos_dict2=position_dict2, only_cnots=only_cnots)
    else:
        #initial entanglers
        U += add_entanglers(r, r_in, m-1, ql_1=ql_1, ql_2=ql_2, pos_dict1=position_dict1, pos_dict2=position_dict2, only_cnots = only_cnots)
    
    U += U1
    U += U2

    position_dict2 = add_key_value_dict(position_dict2, to_add_key=2**(m-1), to_add_value=len(position_dict1.keys()))
    position_dict_new = add_dict(position_dict1, position_dict2)

    return U, position_dict_new

def QRM_rec_punc_circuit(r, m, partitions = [], qubit_list = None, only_cnots = False, state_prep = False, classical=False):
    '''
    Punctured QRM encoder, currently defaulted to dropping first qubit q[0]
    
    '''
    U = tequila.QCircuit()
    punc_bit_list = [0]

    if not state_prep:
        if r >= m or r < m-r-1:
            print('Invalid parameters r, m : {}, {}'.format(r, m))
            return U, None
    else:
        if r > m or r < m-r-1:
            print('Invalid parameters r, m : {}, {}'.format(r, m))
            return U, None
    
    ql_1, ql_2, p1, p2 = get_punc_qubit_partition(m, qubit_list=qubit_list, partitions=partitions, punc_bit_list=punc_bit_list)

    if state_prep:
        #no need 111..11, so can go till m = r + 1
        if (m == r+1) or (m == r):
            if r == 1 and m == 1: #[0 1]^* = [1]
                return U, {1: 0}
            
            U1, position_dict1 = QRM_rec_punc_circuit(m-1, m-1, partitions=p1, qubit_list=ql_1, only_cnots=True, state_prep=state_prep, classical=classical)
            U2, position_dict2 = QRM_rec_classical_circuit(m-1, m-1, partitions=p2, qubit_list=ql_2)

            Gr = Grm(m-1, m-1)[1:]
            targets = [get_set_int(get_eval_set(row, reversed=True)) for row in Gr]
            target_dict = {target: i for i, target in enumerate(targets)}

            #add entanglers, no hadamards
            controls = get_dict_values(position_dict1, targets)
            qm1 = [ql_1[a] for a in controls]
            qm2 = [ql_2[a] for a in targets]
            for a, b in zip(qm1, qm2):
                U += tequila.gates.CX(control=a, target=b)
            #U += add_punc_entanglers(r1=1, r2=m-1, m=m-1, ql_1=ql_1, ql_2=ql_2, pos_dict1=position_dict1, pos_dict2=position_dict2, only_cnots=True, puncture_Gp=False)

            U += U1
            U += U2

            position_dict2 = add_key_value_dict(position_dict2, to_add_key=2**(m-1), to_add_value=len(position_dict1.keys()))
            position_dict_new = add_dict(position_dict1, position_dict2)
            return U, position_dict_new
    else:
        if m == r + 1: #U^*(r, r+1)
            if r == 0: #[1]
                return U, {0: 0}
            
            U1, position_dict = QRM_rec_punc_circuit(r-1, m-1, partitions=p1, qubit_list=ql_1, only_cnots=True, state_prep=state_prep, classical=classical)
            U2, position_dict2 = QRM_rec_classical_circuit(r-1, r, partitions=p2, qubit_list=ql_2)

            Gr = Grm(r-1, r)
            targets = [get_set_int(get_eval_set(row, reversed=True)) for row in Gr]
            target_dict = {target: i for i, target in enumerate(targets)}

            #add entanglers, no hadamards
            controls = get_dict_values(position_dict, targets)
            qm1 = [ql_1[a] for a in controls]
            qm2 = [ql_2[a] for a in targets]
            for a, b in zip(qm1, qm2):
                U += tequila.gates.CX(control=a, target=b)
            
            U += U1
            U += tequila.gates.CX(control=ql_2[-1], target=ql_1[-1])
            U += U2

            target_dict = add_key_value_dict(target_dict, to_add_key=2**r, to_add_value=len(position_dict.keys()))
            position_dict_new = add_dict(position_dict, target_dict)
            position_dict_new[2**r - 1] = 2**(r+1) - 2

            return U, position_dict_new
    
    U1, position_dict1 = QRM_rec_punc_circuit(r, m-1, partitions=p1, qubit_list=ql_1, only_cnots=only_cnots, state_prep=state_prep, classical=classical)
    U2, position_dict2 = QRM_rec_circuit(r, m-1, partitions=p2, qubit_list=ql_2, only_cnots=only_cnots, classical=classical)

    if classical:
        only_cnots=True
        if state_prep:
            r1 = 1
        else:
            r1 = 0
    else:
        r1 = m-r-1
    if r > m-r-1:
        #add message ents
        U += add_punc_entanglers(m-r, r, m-1, ql_1=ql_1, ql_2=ql_2, pos_dict1=position_dict1, pos_dict2=position_dict2, only_cnots=True, puncture_Gp=False)
    
    #add entanglers
    U += add_punc_entanglers(r1, m-r-1, m-1, ql_1=ql_1, ql_2=ql_2, pos_dict1=position_dict1, pos_dict2=position_dict2, only_cnots=only_cnots, puncture_Gp=False)

    U += U1
    U += U2

    position_dict2 = add_key_value_dict(position_dict2, to_add_key=2**(m-1), to_add_value=len(position_dict1.keys()))
    position_dict_new = add_dict(position_dict1, position_dict2)
    
    return U, position_dict_new

def QRM_rec_assym_punc_circuit(r, m, r_in, m_in, partitions = [], qubit_list = None, only_cnots = False, state_prep = False):
    U = tequila.QCircuit()

    if m == 0:
        return U, {0: 0}
    
    if r >= m or r < 0:
        print('Invalid parameters r, m : {}, {}'.format(r, m))
        return U, None
    
    ql_1, ql_2, p1, p2 = get_punc_qubit_partition(m, qubit_list=qubit_list, partitions=partitions)

    if 2*r_in + 1 < m_in:
        if r == 0:
            # need to change!!
            return QRM_rec_punc_circuit(r_in, m, partitions=partitions, qubit_list=qubit_list, only_cnots=only_cnots, state_prep=state_prep, classical=True)
    if 2*r_in + 1 >= m_in:
        if m == r_in + 1:
            Uc, position_dict = QRM_rec_punc_circuit(r_in, r_in + 1, partitions=partitions, qubit_list=qubit_list, only_cnots=only_cnots, state_prep=state_prep)

            #add hadamards
            if 2*r_in + 1 > m_in:
                if not only_cnots:
                    U += add_hadamards(1, 2*r_in - m_in + 1, m, qubit_list=qubit_list, pos_dict=position_dict)
            
            U += Uc
            
            return U, position_dict
    
    U1, position_dict1 = QRM_rec_assym_punc_circuit(r-1, m-1, r_in, m_in, partitions=p1, qubit_list=ql_1, only_cnots=only_cnots, state_prep=state_prep)
    U2, position_dict2 = QRM_rec_assym_circuit(r-1, m-1, r_in, m_in, partitions=p2, qubit_list=ql_2, only_cnots=only_cnots)

    if r != r_in:
        U += add_punc_entanglers(r+1, r_in, m-1, ql_1=ql_1, ql_2=ql_2, pos_dict1=position_dict1, pos_dict2=position_dict2, only_cnots = True, puncture_Gp=False)
        U += add_punc_entanglers(r, r, m-1, ql_1=ql_1, ql_2=ql_2, pos_dict1=position_dict1, pos_dict2=position_dict2, only_cnots=only_cnots, puncture_Gp=False)
    else:
        #initial entanglers
        U += add_punc_entanglers(r, r_in, m-1, ql_1=ql_1, ql_2=ql_2, pos_dict1=position_dict1, pos_dict2=position_dict2, only_cnots = only_cnots, puncture_Gp=False)
    
    U += U1
    U += U2

    position_dict2 = add_key_value_dict(position_dict2, to_add_key=2**(m-1), to_add_value=len(position_dict1.keys()))
    position_dict_new = add_dict(position_dict1, position_dict2)

    return U, position_dict_new