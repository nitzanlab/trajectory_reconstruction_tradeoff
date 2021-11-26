import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr

def get_pairwise_distances_branch(pseudotime, branch, branch_time_dict):
    """
    Computes cell-cell distances from branch-pseudotime structure
    :param pseudotime: pseudotime per cell
    :param branch: branch per cell
    :param branch_time_dict: dictionary branch pair to number of time points
    :return:
        distances(from topology, normalized)
        max dist
    """
    D = np.zeros((len(branch), len(branch)))
    for i in range(len(branch)):
        for j in range(len(branch)):
            if (branch[i], branch[j]) in branch_time_dict:
                D[i, j] = pseudotime[i] + pseudotime[j] - 2 * branch_time_dict[(branch[i], branch[j])]
            else:
                D[i, j] = pseudotime[i] + pseudotime[j] - 2 * min(pseudotime[i], pseudotime[j])
    dmax = np.max(D)
    D = D / dmax
    return D, dmax

def get_pairwise_distances(expr):
    """
    Computes cells geodesic distances as shortest path in the minimal-fully-connected kNN graph
    :param expr: cells expression
    :return:
        distances( normalized)
        max dist
    """
    D = np.inf
    neighbors = 2
    while (np.sum(D) == np.inf):
        neighbors = neighbors + 1
        A = kneighbors_graph(expr, neighbors, mode='distance', metric='euclidean', include_self=True)
        D = dijkstra(csgraph=A, directed=False, return_predecessors=False)
    dmax = np.max(D)
    D = D/dmax
    return D, dmax


def compare_distances(D0, D):
    """
    Compute error(deviation) between distances
    :param D0: true distances
    :param D: predicted distances
    :return:
        mean absolute error (l1)
        mean squared error (l2^2)
        mean squared error of log(1+x)
        mean correlation of distances
    """
    l1 = round(sum(sum(np.abs(np.array(D) - np.array(D0)))) / len(D) ** 2, 3)
    l2 = round(sum(sum(((np.array(D) - np.array(D0)) ** 2))) / len(D) ** 2, 3)
    l3 = round(sum(sum(((np.log(1 + np.array(D)) - np.log(1 + np.array(D0))) ** 2))) / len(D) ** 2, 3)  # rmlse

    lsp = []
    for r0, r in zip(D0, D):
        lsp.append(spearmanr(r0, r))
    lsp = np.mean(lsp)

    return l1, l2, l3, lsp
