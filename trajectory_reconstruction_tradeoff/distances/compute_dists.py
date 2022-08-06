import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pylab as plt
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

def get_pairwise_distances(pX, return_predecessors=False, plot=False, verbose=False):
    """
    Computes cells geodesic distances as shortest path in the minimal-fully-connected kNN graph
    :param pX: cells reduced representation 
    :return:
        distances( normalized)
        max dist
    """
    D = np.inf
    neighbors = 2
    pX = pX.values if isinstance(pX, pd.DataFrame) else pX
    while (np.sum(D) == np.inf):
        neighbors = neighbors + 1
        A = kneighbors_graph(pX, neighbors, mode='distance', metric='euclidean', include_self=True)
        DP = dijkstra(csgraph=A, directed=False, return_predecessors=return_predecessors)
        D = DP[0] if return_predecessors else DP
    
    # dmax = np.max(D) # dmax = np.max(D) # TODO: BIG CHANGE
    # D = D / dmax
    if verbose:
        print(f'Running with {neighbors} neighbors')

    if plot:
        G = nx.from_numpy_matrix(A.todense()>0)
        pos = nx.spring_layout(G)
        edge_weight = list(nx.get_edge_attributes(G,'weight').values())

        pos = {}
        for inode,node in enumerate(G.nodes()):
            pos[node] = [pX[inode,0], pX[inode,1]]
        nx.draw(G,pos, width=edge_weight, node_size=3)
        plt.show()
        
    return DP #, TODO: temp neighbors dmax


def get_path(src, des, P): # TODO: is this a subfunction of below?
    """
    Given (dijkstra's) predecessors matrix, finds the path between src and des.
    """
    path = []
    while (des != src) & (des>=0):
        path.append(des)
        des = P[src][des]

    path.append(src)
    return path



def compute_path_vertex_length(P):
    """
    Given (dijkstra's) predecessors matrix, computes the number of vertices along each path.
    :return: number of vertices along the path
    """
    n = len(P)
    # V = np.full_like(P, np.inf, dtype=np.double)
    V = np.zeros_like(P)
    # right now n^3 so pretty bad...
    for src in range(n):
        for des in np.arange(src, n): # assuming symmetry
            i = src
            
            n_vertices = 0
        
            while (i != des) and (n_vertices < n):
                i = P[des, i]
                n_vertices += 1
            # print(src, des, n_vertices)

            V[src, des] = n_vertices
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


