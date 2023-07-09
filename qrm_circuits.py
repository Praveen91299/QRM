#circuits for QRM codes
#Praveen Jayakumar, July 2023

import tequila
from qrm_utils import *
from qrm_matrices import *

def canonical_CSS(G, qubit_list = None, only_cnot = False):
    '''
    qubit_list is the qubit map. 
    Eg: qubit_list = [2, 3, 5, 6] - the circuits will be over these 4 qubits only
    '''
    Gperp, G1q = G
    n_qubits = len(Gperp[0])
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
        if not only_cnot:
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

def QRM_std_circuit(r, m, qubit_list = None, only_cnot = False):
    params = (r, m)
    Gs = get_QRM_generator(params, params)
    R1 = get_R(Gs[0])#get two 
    R2 = get_R(Gs[1])
    Gperpnew = np.einsum('ap, pb', R1, Gs[0]) % 2
    G1qnew = np.einsum('ap, pb', R2, Gs[1]) % 2
    Gsnew = (Gperpnew, G1qnew)
    #circuit = canonical_CSS(Gs, only_cnot = only_cnot, qubit_list=qubit_list)
    #print(len(circuit.gates))
    circuit, qubit_info = canonical_CSS(Gsnew, only_cnot = only_cnot, qubit_list=qubit_list)
    print(len(circuit.gates))
    return circuit, qubit_info#, msg qubits

#need to generalize stuff
def QRM_rec_circuit(r, m, qubit_list = None):
    U = tequila.QCircuit()

    if qubit_list == None:
        qubit_list = list(range(2**m))
    
    U1, info_1 = QRM_rec_circuit(r, m-1, qubit_list=qubit_list[:2**(m-1)])
    U2, info_2 = QRM_rec_circuit(r, m-1, qubit_list=qubit_list[2**(m-1):])
    info = ()
    #message
    return U, info
