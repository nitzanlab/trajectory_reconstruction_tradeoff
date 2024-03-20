import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pyslingshot import Slingshot
import palantir

##################################################### Utils #####################################################

def get_iroot(adata, idx_col='Trajectory_idx', iroot=None):
    """
    Find index of root cell according to idx_col
    """
    iroot = np.where(adata.obs_names == adata.obs[idx_col].idxmin())[0][0] if iroot is None else iroot
    
    return iroot

def get_iterminal(adata, idx_col='Trajectory_idx', iterminal=None):
    """
    Find index of terminal cell according to idx_col
    """
    iterminal = np.where(adata.obs_names == adata.obs[idx_col].idxmax())[0][0] if iterminal is None else iterminal
    return iterminal


def check_flip_pseudo(pseudo, meta, group_col, group_order, verbose=False):
    """
    Check if the pseudotime is in the right direction
    pseudo - pseudotime
    meta - metadata
    group_col - column in meta that contains milestones
    group_order - order of milestones (for now, assuming linear)
    """
    present_groups = [g for g in group_order if g in meta[group_col].unique()]
    source = present_groups[0]
    sinks = present_groups[-1]
    
    if pseudo[meta[group_col] == source].mean() > pseudo[meta[group_col] == sinks].mean():
        pseudo = -pseudo
        if verbose:
            print('Flipping pseudotime')
    return pseudo

##################################################### Methods #####################################################

def get_dpt(X, meta, group_col, group_order, idx_col='Trajectory_idx', n_neighbors=15, n_pcs=10, n_dcs=10, n_top_genes=100, verbose=False):
    """
    Compute DPT pseudotime [Haghverdi16]
    X - expression
    meta - metadata
    group_col - column in meta that contains milestones
    group_order - order of milestones (for now, assuming linear)
    idx_col - column in meta that contains the original index of the trajectory 
    """
    adata = sc.AnnData(X, obs=meta, dtype=float)
    
    adata.uns['iroot'] = get_iroot(adata, idx_col=idx_col)
    
    sc.pp.recipe_zheng17(adata, n_top_genes=n_top_genes)

    sc.tl.pca(adata, svd_solver="arpack")
    adata.obs['dpt_pseudotime'] = np.inf
    
    while (n_neighbors < adata.shape[0]) and (adata.obs['dpt_pseudotime'] == np.inf).sum() > 0:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata, n_dcs=n_dcs) # , n_branchings=n_branchings
        if verbose:
            print(f'Running with {n_neighbors} neighbors')
        n_neighbors += 1
    
    if verbose:
        print(f'Successfully computed DPT pseudotime with {adata.shape[0]} cells and average of {adata.X.mean().mean():.02f} reads per cell.')
    return adata.obs['dpt_pseudotime'].values


def get_regression(X, meta, group_col, group_order, verbose=False):
    """
    Regression onto start and terminal states [Pusuluri19]
    X - expression
    meta - metadata
    group_col - column in meta that contains the cell type
    group_order - order of cell types (for now, assuming linear)
    """
    
    present_groups = [g for g in group_order if g in meta[group_col].unique()]
    source = present_groups[0]
    sinks = present_groups[-1]
    
    X = np.log1p(X)

    # g - number of genes
    # n - number of cells
    # p - number of types

    # get mean expression of source and sink cell types
    group_X = pd.concat((meta[group_col], X), axis=1)
    group_mean = group_X.groupby(group_col).mean()
    sinks = sinks if isinstance(sinks, list) else [sinks]
    source_exp = group_mean.loc[source]
    sinks_exp = group_mean.loc[sinks]

    # w - a (n x p)
    # A - xsi (g x p)
    # y - S (g x n)
    
    y = X.T # g x n - expression
    A = np.vstack((source_exp, sinks_exp)).T # g x p - cell type mean expression

    # w^T = y^TA(A^TA)^(-1) 
    wT = np.dot(y.T, np.dot(A, np.linalg.pinv(np.dot(A.T,A))))

    S_proj = np.dot(A, wT.T) # g x n - projection of expression on type ab 
    S_perp = (y - S_proj) # remaining expression

    S_perp_norm = np.linalg.norm(S_perp, axis=0)

    a = pd.DataFrame(wT, index=meta.index, columns=[source] + sinks)
    S_perp_norm = pd.Series(S_perp_norm, index=meta.index)

    pseudo = - a[source] / a.sum(axis=1)

    if verbose:
        print(f'Successfully computed regression pseudotime with {X.shape[0]} cells and average of {X.mean().mean():.02f} reads per cell.')

    return pseudo



def get_component1(X, meta, group_col, group_order, verbose=False):
    """
    Get pseudotime ordering based on the first principal component
    """
    X = X.copy()
    meta = meta.copy()
    X = np.log1p(X)
    pseudo = PCA(n_components=1).fit_transform(X)
    pseudo = check_flip_pseudo(pseudo=pseudo, meta=meta, group_col=group_col, group_order=group_order, verbose=verbose)

    if verbose:
        print(f'Successfully computed PC1 pseudotime with {X.shape[0]} cells and average of {X.mean().mean():.02f} reads per cell.')
    return pseudo


def get_paga_pseudotime(X, meta, group_col, group_order, idx_col='Trajectory_idx', n_neighbors=15, n_pcs=10, resolution=1.0, verbose=False, n_top_genes=100):
    """
    Get pseudotime ordering based on PAGA [Wolf19]
    Following tutorial: https://scanpy-tutorials.readthedocs.io/en/latest/paga-paul15.html 
    """
    adata = sc.AnnData(X, obs=meta, dtype=float)
    adata.uns["iroot"] = get_iroot(adata, idx_col=idx_col)

    sc.pp.recipe_zheng17(adata, n_top_genes=n_top_genes)

    sc.tl.pca(adata, svd_solver="arpack")
    adata.obs['dpt_pseudotime'] = np.inf

    while (n_neighbors < adata.shape[0]) and (adata.obs['dpt_pseudotime'] == np.inf).sum() > 0:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.louvain(adata, resolution=resolution)
        sc.tl.paga(adata, groups="louvain")
        sc.tl.dpt(adata)
        if verbose:
            print(f'Running with {n_neighbors} neighbors')
        n_neighbors += 1
    
    if verbose:
        print(f'Successfully computed PAGA pseudotime with {adata.shape[0]} cells and average of {adata.X.mean().mean():.02f} reads per cell.')

    pseudo = adata.obs['dpt_pseudotime'].values
    pseudo = check_flip_pseudo(pseudo=pseudo, meta=meta, group_col=group_col, group_order=group_order, verbose=verbose)
    return pseudo



def get_slingshot_pseudotime(X, meta, group_col, group_order, idx_col='Trajectory_idx', verbose=False):
    """
    Get pseudotime ordering based on SLINGSHOT [Street18]
    """
    adata = sc.AnnData(X, obs=meta)
    sc.pp.recipe_zheng17(adata)
    sc.tl.pca(adata, svd_solver="arpack")
    start_node = 0# get_iroot(adata, idx_col=idx_col)
    slingshot = Slingshot(adata, celltype_key="milestone_id", obsm_key="X_pca", start_node=start_node, debug_level='verbose') # Changed obsm_key=X_umap to obsm_key=X_pca

    slingshot.fit(num_epochs=1)

    if verbose:
        print(f'Successfully computed SLINGSHOT pseudotime with {adata.shape[0]} cells and average of {adata.X.mean().mean():.02f} reads per cell.')
    
    pseudo = slingshot.unified_pseudotime
    pseudo = check_flip_pseudo(pseudo=pseudo, meta=meta, group_col=group_col, group_order=group_order, verbose=verbose)
    return pseudo


def get_palantir_pseudotime(X, meta, group_col, group_order, idx_col='Trajectory_idx', n_top_genes=100, num_waypoints=8, n_neighbors=7, n_components=10, knn=6, verbose=False):
    """
    Get pseudotime ordering based on PALANTIR [Setty19]
    """
    adata = sc.AnnData(X, obs=meta, dtype=float)
    sc.pp.normalize_per_cell(adata)
    palantir.preprocess.log_transform(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="cell_ranger")
    sc.pp.pca(adata)
    # Run diffusion maps
    dm_res = palantir.utils.run_diffusion_maps(adata, n_components=n_components, knn=n_neighbors)
    ms_data = palantir.utils.determine_multiscale_space(adata, n_eigs=2)
    
    n_neighbors = min(adata.shape[0], n_neighbors)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    terminal_states = [meta.iloc[-1].name]
    start_cell = meta.iloc[0].name
    pr_res = palantir.core.run_palantir(adata, start_cell, num_waypoints=num_waypoints, terminal_states=terminal_states, knn=knn)

    if verbose:
        print(f'Successfully computed PALANTIR pseudotime with {adata.shape[0]} cells and average of {adata.X.mean().mean():.02f} reads per cell.')
    
    pseudo = pr_res.pseudotime
    pseudo = check_flip_pseudo(pseudo=pseudo, meta=meta, group_col=group_col, group_order=group_order, verbose=verbose)
    return pseudo