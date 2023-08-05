#utilities for simulation with stim
#Praveen Jayakumar, July 2023
import stim

def stim_CNOT_list(controls, targets):
    circuit = stim.Circuit()
    for c, t in zip(controls, targets):
        circuit.append("CNOT", [c, t])
    return circuit

def stim_H_list(targets = []):
    circuit = stim.Circuit()
    circuit.append("H", targets)
    return circuit

#write stim to tequila and vice versa
def stim_to_tequila(stim_circuit):
    print('Not implemented')
    return

def tequila_to_stim(tq_circuit, noises = []):
    '''
    Converts the tequila circuit into stim circuit, adds noise if noise is not None
    '''
    stim_circuit = stim.Circuit()
    gates = tq_circuit.gates
    for gate in gates:
        if gate.name == 'H':
            stim_circuit += stim_H_list(targets=gate.target)
        if gate.name == 'X':
            stim_circuit += stim_CNOT_list(controls=gate.control, targets=gate.target)
        
        if len(noises) != 0:
            for noise in noises:
                stim_circuit.append("")
    return stim_circuit