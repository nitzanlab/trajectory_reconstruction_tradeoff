import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pylab as plt
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import squareform, pdist

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

def get_pairwise_distances(pX, return_predecessors=False, return_adjacency=False, verbose=False, by_radius=False, radius=None, dim=None):
    """
    Computes cells geodesic distances as shortest path in the minimal-fully-connected kNN graph
    :param pX: cells reduced representation 
    :return:
        distances( normalized)
        max dist
    """
    n = pX.shape[0]
    D = np.inf #np.full((n,n), np.inf)
    pX = pX.values if isinstance(pX, pd.DataFrame) else pX
    
    if by_radius:
        if radius is None:
            radius = 2 * (np.log(n)/n)**(1/(2*dim))
        W = squareform(pdist(pX))
        A = np.zeros((n,n))
        A[W < radius] = W[W < radius]
        DP = dijkstra(csgraph=A, directed=False, return_predecessors=return_predecessors)
        D = DP[0] if return_predecessors else DP
        if verbose:
            print(f'Running with radius {radius}')
            if np.sum(D) == np.inf:
                print('Graph is disconnected')
    
    if radius is None:   
        neighbors = 2
        while (np.sum(D) == np.inf):
            neighbors = neighbors + 1
            A = kneighbors_graph(pX, neighbors, mode='distance', metric='euclidean', include_self=True)
            DP = dijkstra(csgraph=A, directed=False, return_predecessors=return_predecessors)
            D = DP[0] if return_predecessors else DP
        if verbose:
            print(f'Running with {neighbors} neighbors')

    # dmax = np.max(D) # dmax = np.max(D) # TODO: BIG CHANGE
    # D = D / dmax
    if return_adjacency:
        return DP, A        
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


import numpy as np
from scipy.linalg import svd
from sklearn.neighbors import NearestNeighbors

def tangent_subspace(
    X,
    *,
    n_neighbors,
    n_jobs=None,
):
    """Editted from sklearn.manifold Locally Linear Embedding analysis on the data.
    Read more in the :ref:`User Guide <locally_linear_embedding>`.
    Parameters
    ----------
    X : {array-like, NearestNeighbors}
        Sample data, shape = (n_samples, n_features), in the form of a
        numpy array or a NearestNeighbors object.
    n_neighbors : int
        number of neighbors to consider for each point.
    n_jobs : int or None, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    Returns
    -------
    V : left eigenvectors per point N x N x d_in
    eval : eigenvalues per point
        
    """
    N, d_in = X.shape
    n_neighbors = min(n_neighbors, N - 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nbrs.fit(X)
    X = nbrs._fit_X

    # N, d_in = X.shape

    neighbors = nbrs.kneighbors( X, n_neighbors=n_neighbors + 1, return_distance=False)
    neighbors = neighbors[:, 1:]

    V = np.zeros((N, d_in, d_in))
    # V = np.zeros((N, n_neighbors, n_neighbors))
    nev = min(d_in, n_neighbors)
    evals = np.zeros([N, nev])

    # choose the most efficient way to find the eigenvectors
    use_svd = n_neighbors > d_in

    if use_svd:
        for i in range(N):
            X_nbrs = X[neighbors[i]] - X[i] # Changed: want this to be features x neighbors instead of neighbors x features
            V[i], evals[i], _ = svd(X_nbrs.T, full_matrices=True)
    return V, evals




def compute_reach(pX, n_components=6, n_neighbors=10, verbose=False):
    """
    Computes the reach of each cell in the graph.  
    :param pX: cells reduced representation
    :param d: degree of tangent subspace
    :return: \hat{\tau} = \inf_{x,y} \frac{\|x - y\|^2}{2 d(x-y, T_xM)}
    """
    pX = pX.values if isinstance(pX, pd.DataFrame) else pX
    nc,n_dim = pX.shape
    U, _ = tangent_subspace(pX, n_neighbors=n_neighbors) # V holds the U matrix for each point based 
    U = U[:,:,:n_components] # truncate to intrinsic dimension, each U is of shape (n_neighbors, n_dim)
    x_min_y = np.repeat(pX, nc, axis=0).reshape(nc, nc, n_dim) - np.tile(pX, (nc,1)).reshape(nc, nc, n_dim) # nc x nc x n_dim
    norm_x_min_y = np.linalg.norm(x_min_y, axis=2) # nc x nc
    d_x_min_y_TxM = np.zeros((nc, nc)) # nc x nc
    for i in range(nc):
        P = U[i,:,:] @ U[i,:,:].T
        proj_x_min_y = (P @ x_min_y[i].reshape(nc, n_dim).T).T # nc x n_dim
        norm_proj_x_min_y = np.linalg.norm(proj_x_min_y, axis=1) # nc
        d_x_min_y_TxM[i,:] = norm_x_min_y[i,:] - norm_proj_x_min_y # nc - for each point, the distance to the tangent space at point i

    return np.nanmin(norm_x_min_y**2 / (2*d_x_min_y_TxM))


def compute_density(D):
    """
    Computes the density of each cell in the graph.

    :return: min_j d^M_{ij} \leq a, \forall i\in[n_c]
    """
    # should we compute this on the normalized distances
    n = D.shape[0]
    D_max = D.max()
    a = np.max((D + D_max * np.eye(n)).min(axis=1))
    return a