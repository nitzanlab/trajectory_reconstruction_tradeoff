import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist
from collections import deque
import networkx as nx

######################### Edited from https://github.com/mossjacob/pyslingshot ##############################

def get_start_node(adata, milestone_ids, clustering):
    """
    Finds cluster node corresponding to first milestone
    :param adata: scanpy AnnData
    :param milestone_ids: ordered list of milestones
    :param clustering: column name in obs to use for clustering
    :return: start node value
    """
    if clustering == 'milestone_id':
        found = adata.obs[clustering].unique()
        for m in milestone_ids:
            if m in found:
                break
        return m

    cross_counts = adata.obs[[clustering, 'milestone_id']].groupby(['milestone_id', clustering]).size().reset_index()
    for m in milestone_ids:
        tmp = cross_counts[cross_counts['milestone_id'] == m]
        if tmp.shape[0] == 0:
            continue
        else:
            idx = tmp[0].idxmax()
            start_node = tmp.loc[idx][clustering]
            break
    return start_node


def get_tree_children(dists, start_node):
    """
    Computes MST as in pyslingshot
    :param dists: cluster distances (numpy array)
    :param start_node: index of starting node
    :return: dictionary of children of each cluster
    """
    tree = minimum_spanning_tree(dists)
    num_clusters = dists.shape[0]
    connections = {k: list() for k in np.arange(num_clusters)}
    cx = tree.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        connections[i].append(j)
        connections[j].append(i)

    # for i,j,v in zip(cx.row, cx.col, cx.data):
    visited = [False for _ in np.arange(num_clusters)]
    queue = list()
    queue.append(start_node)
    children = {k: list() for k in np.arange(num_clusters)}
    while len(queue) > 0:  # BFS to construct children dict
        current_node = queue.pop()
        visited[current_node] = True
        for child in connections[current_node]:
            if not visited[child]:
                children[current_node].append(child)
                queue.append(child)
    return children


def plot_MST(data, lineages, cluster_centres, cluster_labels, ax=None):
    """
    Plotting MST
    :param data: reduced expression representation (e.g. PCA)
    :param lineages: list of all lineages(all cluster paths)
    :param cluster_centres: cluster centers in reduced expression representation
    :param cluster_labels: cells cluster labels
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    colors = {}
    clusters = cluster_centres.index
    num_clusters = len(clusters)
    cmap = plt.cm.get_cmap('tab20', num_clusters) # TODO: vary depending on size
    for k, i in zip(clusters, range(cmap.N)):
        rgba = cmap(i)
        colors[k] = matplotlib.colors.rgb2hex(rgba)
    ax.scatter(data[:, 0], data[:, 1], c=cluster_labels.map(colors))
    for lineage in lineages:
        prev = None
        for l in lineage:
            if prev is not None:
                start = [cluster_centres.loc[prev][0], cluster_centres.loc[l][0]]
                end = [cluster_centres.loc[prev][1], cluster_centres.loc[l][1]]
                ax.plot(start, end, c='black')
            prev = l


def get_MST_from_lineages(lineages, dists=None):
    """
    Construct graph from lineages
    :param lineages: list of cluster lineages
    :param dists: if given, includes distances between clusters
    :return:
    """
    G = nx.Graph()
    for lineage in lineages:
        prev = None
        for i in lineage:
            if prev is not None:
                length = dists.loc[prev, i] if dists is not None else 1
                G.add_edge(str(prev), str(i), length=length)
            prev = i
    return G



def mahalanobis(X1, X2, S1, S2):
    S_inv = np.linalg.inv(S1 + S2)
    diff = (X1 - X2).reshape(-1, 1)
    return np.matmul(np.matmul(diff.T, S_inv), diff)[0][0]


def get_MST(data, cluster_labels, start_node, metric='euclidean', plot=False):
    """

    :param data:
    :param cluster_labels:
    :param start_node:
    :param with_dists:
    :param plot:
    :return:
    """
    clusters = list(cluster_labels.unique())

    cluster_centres = [data[cluster_labels == k].mean(axis=0) for k in clusters]
    cluster_centres = pd.DataFrame(np.stack(cluster_centres), index=clusters)

    if metric == 'slingshot':
        # Calculate empirical covariance of clusters
        emp_covs = np.stack([np.cov(data[cluster_labels == i].T) for i in clusters])
        dists = pd.DataFrame(0, index=clusters, columns=clusters)
        for i, ci in enumerate(clusters):
            for j, cj in enumerate(clusters[i:]):
                dist = mahalanobis(
                    cluster_centres.loc[ci].values,
                    cluster_centres.loc[cj].values,
                    emp_covs[i],
                    emp_covs[j]
                )
                dists.loc[ci, cj] = dist
                dists.loc[cj, ci] = dist
    else:
        dists = squareform(pdist(cluster_centres, metric=metric))
        dists = pd.DataFrame(dists, index=clusters, columns=clusters)

    start_node_idx = [i for i, c in enumerate(clusters) if c == start_node][0]
    tree = get_tree_children(dists.values, start_node_idx)

    # Determine lineages by parsing the MST
    branch_clusters = deque()

    def recurse_branches(path, v):
        num_children = len(tree[v])
        if num_children == 0:  # at leaf, add a None token
            return path + [v, None]
        elif num_children == 1:
            return recurse_branches(path + [v], tree[v][0])
        else:  # at branch
            branch_clusters.append(v)
            return [recurse_branches(path + [v], tree[v][i]) for i in range(num_children)]

    def flatten(li):
        if li[-1] is None:  # special None token indicates a leaf
            yield li[:-1]
        else:  # otherwise yield from children
            for l in li:
                yield from flatten(l)

    lineages = recurse_branches([], start_node_idx)
    lineages = list(flatten(lineages))
    idx_cluster_dict = {i: c for i, c in enumerate(clusters)}
    lineages = [[idx_cluster_dict[l] for l in lineage] for lineage in lineages]

    if plot:
        plot_MST(data, lineages, cluster_centres, cluster_labels)

    G = get_MST_from_lineages(lineages, dists=dists)
    return G

def get_clusters_MST(adata, milestone_ids, clustering='kmeans', metric='euclidean', plot=False):
    if 'X_pca' not in adata.obsm.keys():
        print('Compute PCA')
        return
    # kmeans clustering
    if clustering == 'kmeans':
        n_milestones = len(milestone_ids)
        kmeans = KMeans(n_clusters=n_milestones, random_state=0).fit(adata.obsm['X_pca'])
        adata.obs[clustering] = kmeans.labels_.astype(str)

    start_node = get_start_node(adata, milestone_ids, clustering)
    G = get_MST(data=adata.obsm['X_pca'], cluster_labels=adata.obs[clustering], start_node=start_node, metric=metric,
                plot=plot)
    return G



if __name__ == '__main__':
    import os
    from trajectory_reconstruction_tradeoff.io import read_data
    from trajectory_reconstruction_tradeoff.opt.opt import subsample

    dirname = '/Users/nomo/PycharmProjects/Tree_Reconstruct_Limitations/datasets/'  # TODO: change to relative path
    fname = 'hayashi'
    fname_counts = os.path.join(dirname, 'counts_' + fname + '.csv')
    fname_dists = os.path.join(dirname, 'geodesic_' + fname + '.csv')
    X, expr_red, D, D_true = read_data(fname_counts=fname_counts, fname_dists=fname_dists)
    meta = pd.read_csv('/Users/nomo/PycharmProjects/Tree_Reconstruct_Limitations/datasets/%s_cell_info.csv' % fname,
                       index_col=1)
    # X, expr_red, D, D_true = simulate("((C:100, D:100)B:100)A:100;")
    adata = sc.AnnData(X, obs=meta)

    p_c = 0.05; p_t = 1
    X1, expr_red1, D0, D1, ix = subsample(X, D, p_c, p_t)  # subsample(X, D_true, p_c, p_t)
    adata1 = sc.AnnData(X1)
    adata1.obs = adata.obs.iloc[ix].copy()

    sc.pp.log1p(adata1)
    sc.pp.pca(adata1, n_comps=10)
    sc.pl.pca(adata1)
    milestones_id = meta['milestone_id'].unique()
    n_milestones = len(milestones_id)
    clustering = 'kmeans'
    kmeans = KMeans(n_clusters=n_milestones, random_state=0).fit(adata1.obsm['X_pca'])
    adata1.obs[clustering] = kmeans.labels_.astype(str)

    start_node = '0' # get_start_node(adata1, milestones_id, 'kmeans')
    G = get_MST(adata1.obsm['X_pca'], adata1.obs['kmeans'], start_node, metric='slingshot', plot=True)
    plt.show()
    nx.draw(G); plt.show()
