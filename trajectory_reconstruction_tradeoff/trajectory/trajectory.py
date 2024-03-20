import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import trajectory_reconstruction_tradeoff as T
from sklearn.decomposition import PCA

import random
random.seed(20)
np.random.seed(20)
from numpy.random import default_rng

rng = default_rng()  # Creates a new random number generator


epsilon = 10e-10

def unique_list(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


class Trajectory():
    """
    Trajectory object
    """

    def __init__(self, X, D=None, meta=None, milestone_network=None, group_col='milestone_id', 
    do_preprocess=True, do_log1p=True,  do_sqrt=False, do_full_locs=False, n_comp=10, name=''):
        """Initialize the tissue using the counts matrix and, if available, ground truth distance matrix.
        X                   -- counts matrix (cells x genes)
        D                   -- ground truth cell-to-cell distances (cells x cells)
        meta                -- other available information per cell (using 'milestone_id' for group orderings)
        milestone_network   -- a milestone network dataframe with columns 'from', 'to' with milestone_id values to indicate milestone graph connections
        group_col           -- column name in meta to use for group ordering
        do_preprocess       -- if False skips complete preprocessing step
        do_log1p            -- if to perform log1p transformation transformation
        do_sqrt             -- if to perform sqrt transformation transformation
        do_full_locs        -- if to use cell locations from full data
        n_comp -- number of components for PCA
        name -- optional saving of dataset name
        """
        # standardize input
        self.ncells, self.ngenes = X.shape
        if isinstance(X, pd.DataFrame):
            genenames = X.columns
            cellnames = X.index
            if meta is not None:
                if (cellnames != meta.index).any():
                    print('Counts and metadata index differ')
                    return
                
        if not isinstance(X, pd.DataFrame):
            genenames = ['g%d' % i for i in range(self.ngenes)]
            cellnames = ['c%d' % i for i in range(self.ncells)] if meta is None else meta.index
            X = pd.DataFrame(X, columns=genenames, index=cellnames)

        if D is not None:
            assert(D.shape[0] == D.shape[1] == self.ncells)
            if not isinstance(D, pd.DataFrame):
                D = pd.DataFrame(D, index=cellnames, columns=cellnames)


        # order data by milestone net
        # linearizing groups
        self.group_order = []
        self.group_col = None
        if group_col in meta.columns:
            self.group_col = group_col
            self.group_order = meta[group_col].unique()
        if milestone_network is not None:
            if isinstance(milestone_network, pd.DataFrame) and ('from' in milestone_network.columns) and ('to' in milestone_network.columns) and ('milestone_id' in meta.columns):    
            
                milestone_ordering = unique_list(list(milestone_network['from'].values) + list(milestone_network['to'].values))
                self.group_order = milestone_ordering
                assert(meta['milestone_id'].isin(milestone_ordering).all())
                dct = {v: i for i, v in enumerate(milestone_ordering)}
                idx = np.argsort(meta['milestone_id'].map(dct), kind='mergesort')
                meta = meta.iloc[idx]
                X = X.iloc[idx]
                cellnames = [cellnames[i] for i in idx]
                if D is not None:
                    D = D.iloc[idx][idx]
        
        self.cellnames = cellnames
        self.genenames = genenames

        # save configs
        self.X = X
        self.do_preprocess = do_preprocess
        self.do_full_locs = do_full_locs
        if do_log1p and do_sqrt:
            ValueError('Should do either log1p or sqrt for preprocess')
        self.do_log1p = do_log1p
        self.do_sqrt = do_sqrt
        self.n_comp = n_comp
        self.pX = None
        self.name = name
    
        # preprocess
        self.pX, self.lX, self.pca = self.preprocess(self.X, return_pca=True)
        
        self.dim = None
        self.P = None 
        if D is None:
            D, n_neighbors, P = T.ds.get_pairwise_distances(self.pX, return_predecessors=True)
            self.P = P

        self.D = D
        self.n_neighbors = n_neighbors
        self.milestone_network = milestone_network
        self.meta = meta if meta is not None else pd.DataFrame(index=cellnames)
        self.meta.loc[:,'Trajectory_idx'] = np.arange(self.ncells)
        
        # for expression analysis
        self.pseudo_df = None
        self.comp_exp_genes = None
        self.n_buckets = None
        self.method_buckets_mean = None

    
    def preprocess(self, X, verbose=False, return_pca=False):
        """
        Standard preprocess
        Optional count transformation (log1p or sqrt),
        followed dimensionality reduction (highly variable genes or PCA)
        X - expression counts (cells x genes)
        return_pca - return PCA object
        
        Returns:
        pX - transformed and reduced expression
        lX - transformed expression
        """
        # return data as is without preprocessing
        pX = X.copy()
        lX = X.copy()
        pca = None
        do_preprocess = self.do_preprocess

        # use the full cell locations (similar to without pp but can be a latent representation)
        if (self.do_full_locs) and (self.pX is not None):
            lX = self.lX.loc[X.index]
            pX = self.pX.loc[X.index]
            pca = self.pca
            do_preprocess = False

        if do_preprocess:
            lX = X.copy()
            # transform
            if self.do_log1p:
                if verbose:
                    print('do_log1p')
                lX = np.log1p(X)
            elif self.do_sqrt:
                if verbose:
                    print('do_sqrt')
                lX = np.sqrt(X)

            # reduce dimensions
            pca = PCA(n_components=self.n_comp)
            pX = pca.fit_transform(lX)
            pcnames = ['PC%d' % (i+1) for i in np.arange(pX.shape[1])]
            pX = pd.DataFrame(pX, index=X.index, columns=pcnames)
    
        if return_pca:
            return pX, lX, pca

        return pX, lX


    def get_genes(self, n_genes=50, perc_top_hvgs=0.1, **kwargs):
        """
        Get genes highly expressed and variable
        n_genes - number of genes to select
        perc_top_hvgs - percentage of top highly variable genes to consider

        Returns:
        hvgs - list of highly variable genes
        """
        n_hvgs = int(self.ngenes * perc_top_hvgs) 
        adata = sc.AnnData(self.X)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, **kwargs)
        adata.var['genename'] = self.X.columns
        ihvgs = np.where(adata.var['highly_variable'])[0]
        hvgs = list(adata.var.iloc[ihvgs]['genename'])
        
        # select genes with high expression
        hvgs_mean = self.X[hvgs].mean()
        n_hhvgs = min(n_genes, len(hvgs)) # min(50, len(hvgs))
        hvgs = list(hvgs_mean.sort_values()[-n_hhvgs:].index)
        return hvgs


    def subsample_counts(self, pc, pt):
        """
        Subsample cells and reads
        X - expression counts
        pc - cell capture probability
        pt - transcript capture probability
        
        Returns:
            subsampled expression
            index of subsampled cells
        """
        n = int(self.ncells * pc)
        if n < self.ncells:
            ix = list(rng.choice(self.cellnames, n, replace=False))
        elif n == self.ncells:
            ix = self.cellnames
        elif n > self.ncells:
            raise ValueError('n > self.ncells')
            
        sX = self.X.loc[ix, :]
        
        cellnames = sX.index
        genenames = sX.columns
        if pt < 1:
            sX = sX.astype(int)
            sX = np.random.binomial(sX, pt)
        sX = pd.DataFrame(sX, index=cellnames, columns=genenames)
        return sX, ix


    def subsample(self, pc, pt, sX=None, ix=None, verbose=False):
        """
        Subsample cells and reads
        pc - cell capture probability
        pt - transcript capture probability
        ix - index of cells to subsample
        
        Returns:sX, psX, lsX, psD, sD, psP, ix, pca
        sX - subsampled expression
        psX - subsampled reduced expression
        lsX - subsampled transformed expression
        psD - subsampled reduced distances
        sD - subsampled full distances
        psP - subsampled reduced predecessors
        ix - index of subsampled cells
        pca - pca transformation
        """
        if ix is None:
            sX, ix = self.subsample_counts(pc, pt)
        else:
            if sX is None:
                sX = self.X.loc[ix, :]


        sD = self.D.loc[ix][ix]  # subsampled ground truth pairwise distances
        sD_max = sD.max().max()
        sD = sD / sD_max

        psX, lsX, pca = self.preprocess(sX, return_pca=True)
        psD, sn_neighbors, psP = T.ds.get_pairwise_distances(psX, return_predecessors=True, verbose=verbose) 
        
        return sX, psX, lsX, psD, sn_neighbors, sD, psP, ix, pca



    def downsample_params(self, B, Pc=None, Pt=None, min_reads=20, verbose=False):
        """
        Filtering downsampling params
        B - sequencing budget
        Pc - cell downsample probabilities
        Pt - read downsample probabilities
        min_reads - minimum number of reads per cell on average
        
        Returns:
        dws_params - dataframe of subsampling experiments
        """
        subsample = 'both'
        if (Pc is None) and (Pt is None):
            ValueError('Need to set Pc or Pt')
        if B is None or B <= 0:
            if Pt is None:
                Pt = 1
                subsample = 'cells'
            if Pc is None:
                Pc = 1
                subsample = 'reads'
            if verbose:
                print('Subsampling only {}'.format(subsample))
        if Pt is None:
            Pt = [B / pc for pc in Pc]
        if Pc is None:
            Pc = [B / pt for pt in Pt]

        dws_params = pd.DataFrame({'pc': Pc, 'pt': Pt})

        if B > 1:
            ValueError('B needs to be smaller than 1')

        min_cells = self.n_comp if self.do_preprocess else 5
        cond = dws_params['pc'] < min_cells / self.ncells
        if np.any(cond):
            if verbose: print(f'Restricting Pc to minimum number of {min_cells} cells')
            dws_params = dws_params[~cond]

        max_cells = self.ncells - 5
        cond = dws_params['pc'] > 1 
        if np.any(cond):
            if verbose: print('Restricting Pc so can subsample')
            dws_params = dws_params[~cond]

        cond = dws_params['pc'] < 0
        if np.any(cond):
            if verbose: print('Restricting Pc to non-negative')
            dws_params = dws_params[~cond]

        cond = dws_params['pc'] < B
        if np.any(cond):
            if verbose: print('Restricting Pc to budget limit')
            dws_params = dws_params[~cond]

        cond = dws_params['pt'] > 1
        if np.any(cond):
            if verbose: print('Restricting Pt to 1')
            dws_params = dws_params[~cond]

        cond = dws_params['pt'] < (min_reads / self.X.sum(1).mean())
        if np.any(cond):
            if verbose: print(f'Restricting Pt to minimum avg of {min_reads} reads per cell')
            dws_params = dws_params[~cond]

        if dws_params.shape[0] == 0:
            ValueError('All params were filtered out')

        return dws_params

    @staticmethod
    def compute_pseudotime(X, meta, group_col, group_order, n_neighbors, verbose=False):
        """
        Compute pseudotime for each method
        X - expression
        meta - metadata
        group_col - column name in meta to use for group ordering
        group_order - order of groups
        n_neighbors - number of neighbors for pseudotime methods
        
        Returns:
        pseudo_df - dataframe of pseudotime values
        """
        pseudo_df = pd.DataFrame(index=meta.index)
        # intervening, setting neighbors to at least 5
        min_n_neighbors = 5
        if n_neighbors < min_n_neighbors and X.shape[0] > min_n_neighbors:
            n_neighbors = min_n_neighbors
            print(f'Setting n_neighbors to {min_n_neighbors}')
        try:
            pseudo_df['regression'] = T.dw.get_regression(X=X, meta=meta, group_col=group_col, group_order=group_order, verbose=verbose)
        except:
            print('Could not compute regression')

        try:
            pseudo_df['component1'] = T.dw.get_component1(X=X, meta=meta, group_col=group_col, group_order=group_order, verbose=verbose)
        except:
            print('Could not compute component1')

        # try:
        #     pseudo_df['palantir_pseudotime'] = T.dw.get_palantir_pseudotime(X=X, meta=meta, group_col=group_col, group_order=group_order, n_neighbors=n_neighbors, verbose=verbose)
        # except:
        #     print('Could not compute palantir_pseudotime')

        try:
            pseudo_df['dpt'] = T.dw.get_dpt(X=X, meta=meta, group_col=group_col, group_order=group_order, n_neighbors=n_neighbors, verbose=verbose)
        except:
            print('Could not compute dpt')

        try:
            pseudo_df['paga_pseudotime'] = T.dw.get_paga_pseudotime(X=X, meta=meta, group_col=group_col, group_order=group_order, n_neighbors=n_neighbors, verbose=verbose)
        except:
            print('Could not compute paga_pseudotime')

        # try:
        #     pseudo_df['slingshot_pseudotime'] = T.dw.get_slingshot_pseudotime(X=X, meta=meta, verbose=verbose)
        # except:
        #     print('Could not compute slingshot_pseudotime')

        if verbose:
            print('Pseudotime methods:', pseudo_df.columns)
            
        return pseudo_df

        

    def evaluate(self, sX, psX, lsX, psD, sn_neighbors, sD, psP, ix, pca, pc, pt, 
                comp_pseudo_corr=False, comp_exp_corr=False, comp_exp_genes=None, verbose=False, plot=False,):
        """
        Computes statistics of downsampled data
        sX - sampled expression
        psX - latent representation of sampled expression
        lsX - transformed expression of sampled expression (log-transformed)
        psD - distances of latent representation of sampled expression
        sn_neighbors - number of neighbors for fully-connected graph
        sD - full distances of sampled data
        psP - predecessors of latent representation of sampled expression
        ix - index of sampled cells
        comp_pseudo_corr - whether to compute pseudotime correlation
        comp_exp_corr - whether to compute expression correlation
        comp_exp_genes - optional list of genes to save bucket expression for
        """
        nc = sX.shape[0]
        nr = sX.sum(1).mean()
        
        smeta = self.meta.loc[ix].copy()

        dmax_psD = psD.max().max(); npsD = psD / dmax_psD
        dmax_sD = sD.max().max(); nsD = sD / dmax_sD

        # compute error
        l1, l2, ldist, fcontrac, fexpand, lsp = T.ds.compare_distances(nsD, npsD)

        
        report = {'nc': nc, 'nr': nr, 'Br': sX.sum().sum(),
                  'l1': l1, 'l2': l2, 'ldist': ldist, 'fcontrac': fcontrac, 'fexpand': fexpand, 'lsp': lsp, 
                  'dmax_psD': dmax_psD, 'dmax_sD': dmax_sD}

        if comp_pseudo_corr or comp_exp_corr:
            spseudo_df = self.compute_pseudotime(X=sX, meta=smeta, group_col=self.group_col, group_order=self.group_order, n_neighbors=sn_neighbors, verbose=verbose)
            for method in spseudo_df.columns:
                corr = np.corrcoef(self.pseudo_df.loc[ix, method], spseudo_df[method])[0, 1]
                report[method + '_corr'] = corr
            
        if comp_exp_corr:
            for method in spseudo_df.columns:
                print(method)
                try:
                    ordered_s_buckets_mean = T.dw.get_mean_bucket_exp(self.X.loc[ix, self.comp_exp_genes], self.pseudo_df.loc[ix, method], n_buckets=self.n_buckets, plot=plot)
                    ordered_exp_corr = T.dw.expression_correlation(self.method_buckets_mean[method], ordered_s_buckets_mean)
                    report[method + '_ordered_exp_corr'] = ordered_exp_corr

                    s_buckets_mean = T.dw.get_mean_bucket_exp(sX[self.comp_exp_genes], spseudo_df[method], n_buckets=self.n_buckets, plot=plot)
                    exp_corr = T.dw.expression_correlation(self.method_buckets_mean[method], s_buckets_mean)
                    report[method + '_exp_corr'] = exp_corr
                except:
                    print(f'Could not compute expression correlation for {method}')

        # report bucket expression for each gene in comp_exp_genes with each method
        if comp_exp_genes is not None:
            comp_exp_genes = list(set(comp_exp_genes).intersection(self.X.columns))
            for method in spseudo_df.columns:
                try:
                    ordered_s_buckets_mean = T.dw.get_mean_bucket_exp(self.X.loc[ix,comp_exp_genes], self.pseudo_df.loc[ix, method], n_buckets=self.n_buckets, plot=plot)
                    for gene in comp_exp_genes:
                        for b in np.arange(self.n_buckets):
                            report[f'{method}_{gene}_{b}_ordered_exp'] = ordered_s_buckets_mean[gene].values[b]

                    s_buckets_mean = T.dw.get_mean_bucket_exp(sX[comp_exp_genes], spseudo_df[method], n_buckets=self.n_buckets, plot=plot)
                    for gene in comp_exp_genes:
                        for b in np.arange(self.n_buckets):
                            report[f'{method}_{gene}_{b}_exp'] = s_buckets_mean[gene].values[b]
                except:
                    print(f'Could not compute expression pattern for genes {comp_exp_genes}, {method}')
            
        return report


    def compute_tradeoff(self, B, Pc=None, Pt=None, repeats=50, verbose=False, plot=False,
                        comp_pseudo_corr=False, comp_exp_corr=False, comp_exp_genes=None, comp_pseudo_gt=None,
                        n_buckets=5, **kwargs):
        """
        Compute reconstruction error for subsampled data within budget opt
        X - counts data
        D_true - geodesic distances
        B - counts budget
        Pc - cell capture probabilities
        repeats - number of repeats
        verbose - print messages
        comp_exp_genes - optional list of genes to save bucket expression for
        plot - plot reduced expression for each set of downsample params
        
        Returns:
            dataframe with sampling params and errors
        """

        dws_params = self.downsample_params(B, Pc, Pt, verbose)

        # pre-compute for entire data
        if comp_pseudo_corr or comp_exp_corr:
            # for all methods, set meta[comp_pseudo_gt] as ground truth pseudotime
            if comp_pseudo_gt is not None:
                if comp_pseudo_gt in self.meta.columns:
                    methods = ['regression', 'component1', 'palantir_pseudotime', 'dpt', 'paga_pseudotime', 'slingshot_pseudotime']
                    self.pseudo_df = pd.DataFrame(index=self.meta.index)
                    for m in methods:
                        self.pseudo_df[m] = self.meta[comp_pseudo_gt]
                else:
                    ValueError(f'No column {comp_pseudo_gt} in meta')
            else:    
                # compute with each method a "ground truth" pseudotime
                self.pseudo_df = self.compute_pseudotime(self.X, self.meta, self.group_col, self.group_order, n_neighbors=self.n_neighbors, verbose=verbose)

        if comp_exp_corr:
            # select genes for reconstruction evaluation
            if self.comp_exp_genes is None:

                self.comp_exp_genes = self.get_genes()

                if verbose:
                    print('Using %d genes' % len(self.comp_exp_genes))
            self.n_buckets = n_buckets

            self.method_buckets_mean = {}
            for method in self.pseudo_df.columns:
                try:
                    self.method_buckets_mean[method] = T.dw.get_mean_bucket_exp(self.X[self.comp_exp_genes], self.pseudo_df[method], n_buckets=self.n_buckets, plot=plot)
                except:
                    print(f'Could not compute expression pattern for {method}')

        # evaluate subsampled data
        L = []
        for _, row in dws_params.iterrows():
            pc = row['pc']
            pt = row['pt']
            for k in range(repeats):
                if verbose:
                    print(k)

                # sample
                subsample_result = self.subsample(pc, pt, verbose=verbose)
                
                report = self.evaluate( *subsample_result, pc=pc, pt=pt, 
                                        comp_pseudo_corr=comp_pseudo_corr, comp_exp_genes=comp_exp_genes, 
                                        comp_exp_corr=comp_exp_corr, verbose=verbose, plot=plot, **kwargs)
                         
                report = {'pc': pc, 'pt': pt, 'B': B, 
                          'log pc': np.log(pc), 'log pt': np.log(pt),  
                          **report}
                
                L.append(report)

        L = pd.DataFrame(L)
        return L

