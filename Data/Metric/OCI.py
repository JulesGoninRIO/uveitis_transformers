import numpy as np
import os
import json

TT_labels = json.load(open(os.path.dirname(__file__)+'/../TT_labels.json'))

_beta = 0.75
_gamma = 2

def OCI_from_data(item, preds, labels, beta=_beta, gamma=_gamma):
    
    names = TT_labels[item]
    n = len(names)
    
    labels = [names.index(label) for label in labels]
    preds = [names.index(pred) for pred in preds]
    confusion_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            confusion_matrix[i, j] = sum(1 for x, y in zip(labels, preds) if (x, y)==(i, j))
            
    return OCI(confusion_matrix, beta=beta, gamma=gamma)

def OCI(M, beta=_beta, gamma=_gamma):
    
    k = len(M)
    beta /= (k-1)*np.sum(M)
    M2 = np.zeros(M.shape)
    for i in range(k):
        for j in range(k):
            M2[i, j] = M[i, j]*abs(i-j)**gamma
    N = np.sum(M)+np.sum(M2)**(1/gamma)
    
    def oci_cost(path, i, j):
        return path[:-1]+[(i, j), path[-1]-M[i, j]/N+beta*M2[i, j]]
    paths = [oci_cost([1], 0, 0)]
    oci = 1
    
    while paths:
        new_paths = []
        for path in paths:
            (i, j) = path[-2]
            if i == j == (k-1):
                oci = min(oci, path[-1])
            if i < k-1:
                new_paths.append(oci_cost(path, i+1, j))
            if j < k-1:
                new_paths.append(oci_cost(path, i, j+1))
            if i<k-1 and j<k-1:
                new_paths.append(oci_cost(path, i+1, j+1))
        paths = new_paths
        
    return oci