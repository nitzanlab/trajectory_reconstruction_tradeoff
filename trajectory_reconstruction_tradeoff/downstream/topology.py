import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

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
    adata.uns['iroot'] = iroot

    
    while True:
        try:
            sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors, method='gauss')
        except ValueError:
            if n_neighbors == 50:
                print('looking')
            n_neighbors += 1 
            continue
            
        if n_neighbors == 50:
            print('looking')
        sc.tl.diffmap(adata, n_comps=10)
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
