import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra


def get_pairwise_distances(pX, return_predecessors=False, return_adjacency=False, verbose=False):
    """
    Computes cells geodesic distances as shortest path in the minimal-fully-connected kNN graph
    :param pX: cells reduced representation
    :param return_predecessors: whether to return predecessors matrix
    :param return_adjacency: whether to return adjacency matrix
    :return:
    distances: normalized geodesic distances
    predecessors: predecessors matrix
    adjacency: adjacency matrix
    """
    n = pX.shape[0]
    D = np.inf 
    index = pX.index if isinstance(pX, pd.DataFrame) else np.arange(n)
    pX = pX.values if isinstance(pX, pd.DataFrame) else pX
    
    neighbors = 2
    while (np.sum(D) == np.inf):
        neighbors = neighbors + 1
        A = kneighbors_graph(pX, neighbors, mode='distance', metric='euclidean', include_self=True)
        D = dijkstra(csgraph=A, directed=False, return_predecessors=False)
    if verbose:
        print(f'Running with {neighbors} neighbors')
    if verbose: 
        print(f'Average number of neighbors: {(A > 0).sum(1).mean()}')

    D = pd.DataFrame(D, index=index, columns=index)
    res = [D]
    if return_predecessors:
        P = dijkstra(csgraph=A, directed=False, return_predecessors=True)[1]
        P = pd.DataFrame(P, index=index, columns=index)
        res.append(P)
    if return_adjacency:
        A = pd.DataFrame(A.todense(), index=index, columns=index)
        res.append(A)
    return res
