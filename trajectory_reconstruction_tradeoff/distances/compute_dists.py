import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra

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
    # dmax = np.max(D) # TODO: BIG CHANGE
    # D = D / dmax
    return D

def get_pairwise_distances(expr_red, return_predecessors=False):
    """
    Computes cells geodesic distances as shortest path in the minimal-fully-connected kNN graph
    :param expr_red: cells expression
    :return:
        distances( normalized)
        max dist
    """
    D = np.inf
    neighbors = 2
    while (np.sum(D) == np.inf):
        neighbors = neighbors + 1
        A = kneighbors_graph(expr_red, neighbors, mode='distance', metric='euclidean', include_self=True)
        DP = dijkstra(csgraph=A, directed=False, return_predecessors=return_predecessors)
        D = DP[0] if return_predecessors else DP
    
    # dmax = np.max(D) # dmax = np.max(D) # TODO: BIG CHANGE
    # D = D / dmax
    return DP #, dmax

def compute_path_vertex_length(P):
    """
    Given dijkstra's predecessors matrix, computes the number of vertices along each path.
    :return:
    """
    n = len(P)
    # V = np.full_like(P, np.inf, dtype=np.double)
    V = np.zeros_like(P)
    # right now n^3 so pretty bad...
    for i1 in range(n):
        for i2 in np.arange(i1, n): # assuming symmetry
            i = i1
            n_vertices = 0
        
            while (i != i2) and (n_vertices < n):
                i = P[i2, i]
                n_vertices += 1
            # print(i1, i2, n_vertices)
            V[i1, i2] = n_vertices
    V = V + V.T
    return V

def dist_by_graph(G, cluster_labels):
    """

    """
    G_dists = graph_to_dists(G)
    one_hot = pd.get_dummies(cluster_labels)
    cells_dists = np.dot(one_hot, np.dot(G_dists.loc[one_hot.columns][one_hot.columns], one_hot.T))
    return cells_dists


# def dist_by_graph(G1, cluster_labels1, G2, cluster_labels2):
#     """
#
#     """
#     if np.any(cluster_labels1.index != cluster_labels2.index):
#         print('Cluster label indices have to correspond')
#         return
#     G1_dists = graph_to_dists(G1)
#     one_hot1 = pd.get_dummies(cluster_labels1)
#     cells_dists1 = np.dot(one_hot1, np.dot(G1_dists.loc[one_hot1.columns][one_hot1.columns], one_hot1.T))
#
#     G2_dists = graph_to_dists(G2)
#     one_hot2 = pd.get_dummies(cluster_labels2)
#     cells_dists2 = np.dot(one_hot2, np.dot(G2_dists.loc[one_hot2.columns][one_hot2.columns], one_hot2.T))
#
#     return compare_distances(cells_dists1, cells_dists2)


def graph_to_dists(G):
    """

    :param G:
    :return:
    """
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    nodes = list(G.nodes)
    G_dists = pd.DataFrame(np.inf, index=nodes, columns=nodes)
    for u in nodes:
        for v in nodes:
            G_dists.loc[u, v] = lengths[u][v]
    G_dists /= G_dists.max()
    return G_dists


