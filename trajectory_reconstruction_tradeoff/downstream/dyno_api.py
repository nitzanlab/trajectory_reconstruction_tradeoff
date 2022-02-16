import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

def graph_to_milestone_network(G):
    """
    Graph representation for dyno (dynverse) trajectory representation
    """
    milestone_list = []
    for e in G.edges:
        milestone_list.append({'from': e[0], 'to': e[1], 'length': G.edges[e]['length']})
    milestone_network = pd.DataFrame(milestone_list)
    milestone_network["directed"] = 'TRUE'
    return milestone_network

def graph_from_milestone_network(milestone_network):
    """
    """
    G = nx.Graph()
    for irow, row in milestone_network.iterrows():
        G.add_edge(row['from'], row['to'], length=row['length'])
    milestone_ids = np.unique(milestone_network[['from', 'to']].values.flatten())
    return G, milestone_ids


def graph_clusters_to_dyno(G, adata, clustering='kmeans', plot=False):
# def graph_clusters_to_dyno(G, adata, milestone_ids, clustering='kmeans', metric='euclidean', plot=False):
    """
    Computes parameters for dyno object
    :param adata: expression
    :param milestone_ids: ordered list of milestones
    :return:
        milestone_network -
        branches -
        branch_network -
        branch_progressions -
        cell_ids -
    """
    # G = get_clusters_MST(adata, milestone_ids, clustering=clustering, metric=metric, plot=plot)
    sc.pp.neighbors(adata)
    sc.tl.diffmap(adata)
    start_id = adata.obs_names[0]
    adata.uns['iroot'] = np.where(adata.obs.index == start_id)[0][0]
    sc.tl.dpt(adata, n_dcs=min(adata.obsm['X_diffmap'].shape[1], 10))

    # grouping
    grouping = pd.DataFrame({"cell_id": adata.obs.index, "group_id": adata.obs.kmeans})

    # milestone network

    milestone_network = graph_to_milestone_network(G)

    # branch progressions: the scaled dpt_pseudotime within every cluster
    branch_progressions = adata.obs
    branch_progressions["dpt_pseudotime"] = branch_progressions["dpt_pseudotime"].replace([np.inf, -np.inf],
                                                                                          1)  # replace unreachable pseudotime with maximal pseudotime
    branch_progressions["percentage"] = branch_progressions.groupby(clustering)["dpt_pseudotime"].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())).fillna(0.5)
    branch_progressions["cell_id"] = adata.obs.index
    branch_progressions["branch_id"] = branch_progressions[clustering].astype(np.str)
    branch_progressions = branch_progressions[["cell_id", "branch_id", "percentage"]]

    # branches:
    # - length = difference between max and min dpt_pseudotime within every cluster
    # - directed = not yet correctly inferred
    branches = adata.obs.groupby(clustering).apply(
        lambda x: x["dpt_pseudotime"].max() - x["dpt_pseudotime"].min()).reset_index()
    branches.columns = ["branch_id", "length"]
    branches["branch_id"] = branches["branch_id"].astype(np.str)
    branches["directed"] = True

    # branch network: determine order of from and to based on difference in average pseudotime
    branch_network = milestone_network[["from", "to"]]
    average_pseudotime = adata.obs.groupby(clustering)["dpt_pseudotime"].mean()
    for i, (branch_from, branch_to) in enumerate(zip(branch_network["from"], branch_network["to"])):
        if average_pseudotime[branch_from] > average_pseudotime[branch_to]:
            branch_network.at[i, "to"] = branch_from
            branch_network.at[i, "from"] = branch_to

    cell_ids = adata.obs.index

    if plot:
        sc.pl.pca(adata, color=[clustering, 'dpt_pseudotime'])

    return milestone_network, branches, branch_network, branch_progressions, cell_ids
