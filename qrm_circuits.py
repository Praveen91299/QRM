#circuits for QRM codes
#Praveen Jayakumar, July 2023

import tequila
from qrm_utils import *
from qrm_matrices import *

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

def QRM_rec_circuit(r, m, partitions = [], qubit_list = None, only_cnots = False):
    U = tequila.QCircuit()
    msg_q = []
    ent_q = []
    if m == 0:
        return U, (qubit_list, [])
    if qubit_list == None:
        qubit_list = list(range(2**m))
    
    if partitions != []:
        ql_1, ql_2 = apply_qubit_partition(partitions[0], qubit_list)
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
    
    if r == m:
        for a, b in zip(ql_1, ql_2):
            U += tequila.gates.CX(control=a, target=b)
        U1, info_1 = QRM_rec_circuit(r-1, m-1, partitions=p1, qubit_list=ql_1, only_cnots=only_cnots)
        U2, info_2 = QRM_rec_circuit(r-1, m-1, partitions=p2, qubit_list=ql_2, only_cnots=only_cnots)
        U = U + U1 + U2
        return U, (qubit_list, info_1[1] + info_2[1])
    
    U1, info_1 = QRM_rec_circuit(r, m-1, partitions=p1, qubit_list=ql_1, only_cnots=only_cnots)
    U2, info_2 = QRM_rec_circuit(r, m-1, partitions=p2, qubit_list=ql_2, only_cnots=only_cnots)
    if r > m-r-1:
        #add message ents
        Gm = get_QRM_generators_r1r2(m-r, r, m-1)
        lbs = [leading_bit_index(row) for row in Gm]
        qm1 = [ql_1[i] for i in lbs]
        qm2 = [ql_2[i] for i in lbs]
        for a, b in zip(qm1, qm2):
            U += tequila.gates.CX(control=a, target=b)
        msg_q += qm1
    
    Gent = get_QRM_generators_r1r2(m-r-1, m-r-1, m-1)
    lbs = [leading_bit_index(row) for row in Gent]
    qent1 = [ql_1[i] for i in lbs]
    qent2 = [ql_2[i] for i in lbs]
    if not only_cnots:
        U += tequila.gates.H(target=qent1)
    for a, b in zip(qent1, qent2):
        U += tequila.gates.CX(control=a, target=b)

    U += U1
    U += U2

    ent_q += qent1 + info_1[1] + info_2[1]
    msg_q = [leading_bit_index(row) for row in get_QRM_generators_r1r2(m-r, r, m)] #This is a placeholder. Need to figure out message bit pass recursively, should be msg_q + something
    return U, (msg_q, ent_q)