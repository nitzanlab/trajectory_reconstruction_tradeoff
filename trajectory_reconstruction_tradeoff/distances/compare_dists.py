import numpy as np
import pandas as pd

def compare_distances(D0, D, verbose=False, eps = 10e-5):
    """
    Compute error(deviation) between distances
    D0 - original distances
    D - reconstructed distances

    Returns:
        mean absolute error (l1)
        mean squared error (l2^2)
        mean squared error of log(1+x)
        mean correlation of distances
        fraction of distances contracted
        fraction of distances expanded
        
    """
    if isinstance(D0, pd.DataFrame) and isinstance(D, pd.DataFrame):
        overlap = D0.index.intersection(D.index)
        if verbose:
            print(f'Computing distances over {len(overlap)} corresponding cells')
        D0 = D0.loc[overlap, overlap]
        D = D.loc[overlap, overlap]
        D0 = D0.values
        D = D.values

    l1 = round(sum(sum(np.abs(np.array(D) - np.array(D0)))) / len(D) ** 2, 3) 
    l2 = round(sum(sum(((np.array(D) - np.array(D0)) ** 2))) / len(D) ** 2, 3)

    ind = np.triu_indices_from(D0, 1)
    lcontrac = D0[ind] / (D[ind] + eps)
    lexpand = D[ind] / (D0[ind] + eps)
    ldist = np.mean(np.maximum(lcontrac, lexpand))
    fcontrac = np.sum(lcontrac > 1) / len(lcontrac)
    fexpand = np.sum(lexpand > 1) / len(lexpand)
    lsp = 0
    
    return l1, l2, ldist, fcontrac, fexpand, lsp
