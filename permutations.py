from tequila import QCircuit

class Permutation:
    """
    Class to describe permutations.

    Stored as dictionaries, with values indicating the positions that the keys are permuted to

    """
    def __init__(self, mapping):
        self._mapping = mapping
        return
    
    @classmethod
    def init_from_vector(cls, vec):
        mapping = {i: vec[i] for i in range(len(vec))}
        return Permutation(mapping)
    
    @property
    def mapping(self):
        return self._mapping
    
    @mapping.setter
    def mapping(self, mapping):
        self._mapping = mapping
    
    def get_matrix(self):
        P = np.zeros((1, 1))
        return P
    
    def permute(self, i):
        return self.mapping[i]
    
    def inv_permute(self, i):
        return list(self.keys())[list(self.values()).index(i)]
    
    def get_inverse(self):
        return Permutation({v: k for k, v in zip(self.keys(), self.values())})
    
    def keys(self):
        return self.mapping.keys()
    
    def values(self):
        return self.mapping.values()
    
    def add_entry(self, k, v):
        self.mapping[k] = v
        return
    
    def __repr__(self) -> str:
        string = "Qubit permutation defined by"
        for k, v in zip(self.keys(), self.values()):
            string += '\n' + str(k) + ' -> ' + str(v)
        return string
    
    def __add__(self, other):
        """
        Adds the permutations by combining the keys and values
        """
        new_mapping = dict(self.mapping)
        new_mapping.update(other.mapping)
        return Permutation(new_mapping)
    
    def __mul__(self, other):
        """
        Combines permutations as self[other[.]]
        """
        new_mapping = dict(other.mapping)
        new_mapping = {k: self.permute(v) for k, v in zip(new_mapping.keys(), new_mapping.values())}
        return Permutation(new_mapping)
    
    def permute_circuit(self, circuit):
        """
        Returns circuit with permuted qubits

        P^{-1} C P

        When written as unitaries: U_p U_c U_p^\dagger 
        """
        if isinstance(circuit, QCircuit):
            return circuit.map_qubits(self.mapping)
        else:
            raise "Unsupported circuit type!"
        return
    
    def fill(self, qubit_list):
        ukeys, uvalues = [], []
        for q in qubit_list:
            if q not in self.keys():
                ukeys.append(q)
            if q not in self.values():
                uvalues.append(q)
        
        ukeys = sorted(ukeys)
        uvalues = sorted(uvalues)
        for k, v in zip(ukeys, uvalues):
            self.add_entry(k, v)
        return

def concatenate_permutations(perm_list, vec_list):
    """
    Concatenate disjoint permutations laterally in perm_list
    on qubits specified by vectors in qubit_vec_list
    """
    new_dict = {}
    for perm, vec in zip(perm_list, vec_list):
        for k, v  in zip(perm.keys(), perm.values()):
            new_dict[vec[k]] = vec[v]
    return Permutation(new_dict)

def combine_permutations(perm_list):
    """
    Combines permutations with the same support
    perm_list = [perm1, perm2, perm3], then
    return Permutation({i: perm3[perm2[perm1[i]]]})
    """
    support = sorted(list(perm_list[0].keys()))
    new_dict = {i: i for i in support}
    for perm in perm_list:
        new_dict = {k: perm.permute(v) for k, v in zip(new_dict.keys(), new_dict.values())}
    return Permutation(new_dict)

def fill_permutation(perm):
    """
    Completes a permutation by adding empty elements upto max element

    """
    n_qubits = len(perm.keys())
    n_max = max(list(perm.keys()) + list(perm.values())) + 1
    extra_keys = [i for i in range(n_max) if i not in perm.keys()]
    extra_values = [i for i in range(n_max) if i not in perm.values()]
    
    mapping = perm.mapping
    for k, v in zip(extra_keys, extra_values):
        mapping[k] = v
    
    return mapping