# from .mst import get_MST, get_start_node
import scanpy as sc
import numpy as np
import pandas as pd

# def topology(X, D, expr_red, meta=None, by='kmeans', n_clusters=None, plot=False, start_node=None):
#     start_node = get_start_node(adata, milestone_ids, by) if start_node is None else start_node
#     Gfull = get_MST(expr_red, cluster_labels=meta[by], start_node=start_node, plot=plot)
#     traj_full = graph_to_milestone_network(Gfull)
#     return traj_full

def get_pseudo_from_adata(adata, use_rep='X_pca', idx_col='Trajectory_idx', n_neighbors=15, verbose=True, iroot=None):
    """
    Compute pseudotime
    adata - AnnData object
    use_rep -
    idx_col - colname in meta that contains the true(or proxy) ordering (for defining root)
    n_neighbors - starting number of neighbors
    """
    n_neighbors = min(adata.shape[0], n_neighbors)

    while True:
        try:
            sc.pp.neighbors(adata, use_rep=use_rep, method='gauss', n_neighbors=n_neighbors)
        except ValueError:
            print(f'Failed to compute neighbors for psuedotime ordering')
            print(f'with n cells: {adata.shape[0]}, neighbors: {n_neighbors}, features: {adata.obsm[use_rep].shape[1]}, total reads: {adata.X.sum()}')
            
            return
        sc.tl.diffmap(adata)
    #     adata.obs['original_idx'] = np.arange(adata.shape[0])
        adata.uns['iroot'] = np.where(adata.obs_names == adata.obs[idx_col].idxmin())[0][0] if iroot is None else iroot
        sc.tl.dpt(adata)
        if (adata.obs['dpt_pseudotime'] == np.inf).sum() > 0:
            n_neighbors += 1
            if verbose:
                print('Disconnected graph, increasing number of neighbors to %d' % n_neighbors)
        else:
            break
    return adata.obs['dpt_pseudotime']


def get_pseudo(X, meta, pX=None, **kwargs):
    """
    Compute pseudotime
    X - expression
    meta - metadata
    idx_col - colname in meta that contains the true(or proxy) ordering (for defining root)
    n_neighbors - starting number of neighbors
    """
    # computing pseudotime with dpt
    adata = sc.AnnData(X)
    adata.obs = meta.loc[X.index]

    if pX is not None:
        use_rep = 'X_user'
        adata.obsm[use_rep] = pX
    else:
        use_rep = 'X_pca'
        sc.pp.log1p(adata)
        sc.tl.pca(adata) # recomputing since 10 pcs give disconnected pseudotime

    return get_pseudo_from_adata(adata, use_rep=use_rep, **kwargs)


def get_pseudo_bucket(X, meta, n_buckets=10, idx_col='Trajectory_idx', **kwargs):
    """
    Computes groups of pseudotime
    X - expression
    meta - metadata
    n_buckets - number of groups
    idx_col - colname in meta that contains the true(or proxy) ordering (for defining root)
    """
    pseudo = get_pseudo(X, meta, idx_col=idx_col, **kwargs)
    if pseudo is not None:
        return pd.qcut(pseudo, q=n_buckets, labels=np.arange(n_buckets))
    else:
        return None
