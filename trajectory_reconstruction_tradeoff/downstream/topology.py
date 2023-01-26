# from .mst import get_MST, get_start_node
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# def topology(X, D, expr_red, meta=None, by='kmeans', n_clusters=None, plot=False, start_node=None):
#     start_node = get_start_node(adata, milestone_ids, by) if start_node is None else start_node
#     Gfull = get_MST(expr_red, cluster_labels=meta[by], start_node=start_node, plot=plot)
#     traj_full = graph_to_milestone_network(Gfull)
#     return traj_full

def get_pseudo_from_adata(adata, use_rep='X_pca', idx_col='Trajectory_idx', n_neighbors=15, verbose=True, iroot=None, plot=False):
    """
    Compute pseudotime
    adata - AnnData object
    use_rep -
    idx_col - colname in meta that contains the true(or proxy) ordering (for defining root)
    n_neighbors - starting number of neighbors
    """
    n_neighbors = min(adata.shape[0], n_neighbors)
    iroot = np.where(adata.obs_names == adata.obs[idx_col].idxmin())[0][0] if iroot is None else iroot
    # order adata so root is first
    # idx_org = adata.obs_names
    # adata_ = sc.concat((adata[iroot], adata[~adata.obs_names.isin([adata.obs_names[iroot]])]))
    adata.uns['iroot'] = iroot

    # sc.tl.dpt(adata, n_neighbors=n_neighbors, knn=False)
    
    while True:
        try:
            sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors, method='gauss')
        except ValueError:
            if n_neighbors == 50:
                print('looking')
            n_neighbors += 1 
            continue
            # print(f'Failed to compute neighbors for psuedotime ordering')
            # print(f'with n cells: {adata.shape[0]}, neighbors: {n_neighbors}, features: {adata.obsm[use_rep].shape[1]}, total reads: {adata.X.sum()}')
            
            # return
        if n_neighbors == 50:
            print('looking')
        sc.tl.diffmap(adata, n_comps=10)
    #     adata.obs['original_idx'] = np.arange(adata.shape[0])
        
        # adata.obs['dpt_pseudotime'] = adata.obsm['X_diffmap'][:,1]
        sc.tl.dpt(adata)
        
        if (adata.obs['dpt_pseudotime'] == np.inf).sum() > 0:
            n_neighbors += 1
            if verbose:
                print('Disconnected graph, increasing number of neighbors to %d' % n_neighbors)
        else:
            break
        
    if plot:
        rep = adata.obsm[use_rep] if isinstance(adata.obsm[use_rep], np.ndarray) else adata.obsm[use_rep].values
        plt.scatter(rep[:,0], rep[:,1], c=adata.obs['dpt_pseudotime'])
        plt.scatter(rep[adata.uns['iroot'],0], rep[adata.uns['iroot'],1], c='r', marker='x')
        plt.show()

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


# def get_pseudo_bucket(X, meta, n_buckets=10, idx_col='Trajectory_idx', plot=False, **kwargs):
#     """
#     Computes groups of pseudotime
#     X - expression
#     meta - metadata
#     n_buckets - number of groups
#     idx_col - colname in meta that contains the true(or proxy) ordering (for defining root)
#     """
#     pseudo = get_pseudo(X, meta, idx_col=idx_col, **kwargs)
#     binned_pseudo = None
#     if pseudo is not None:
#         binned_pseudo = pd.qcut(pseudo, q=n_buckets, labels=np.arange(n_buckets))
#         if plot:
#             plt.scatter(X.iloc[:,0], X.iloc[:,1], c=binned_pseudo)