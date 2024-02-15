import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import trajectory_reconstruction_tradeoff as T
from sklearn.decomposition import PCA

epsilon = 10e-10

def unique_list(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


class Trajectory():
    """
    Trajectory object
    """

    def __init__(self, X, D=None, meta=None, milestone_network=None, group_col='milestone_id', outdir=None, 
    do_preprocess=True, do_log1p=True,  do_sqrt=False, do_original_locs=False, n_comp=10, do_hvgs=False, n_hvgs=100, name=''):
        """Initialize the tissue using the counts matrix and, if available, ground truth distance matrix.
        X      -- counts matrix (cells x genes)
        D      -- if available, ground truth cell-to-cell distances (cells x cells)
        meta   -- other available information per cell
        outdir  -- folder path to save the plots and data
        do_preprocess -- if False skips complete preprocessing step
        do_log1p -- if to perform log1p transformation transformation
        do_sqrt -- if to perform sqrt transformation transformation
        do_original_locs -- if to use original("true") cell locations
        do_hvgs -- if True, use highly variable genes to reduce the dimensionality of the expression matrix
        n_hvgs -- number of highly variable genes to use
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
        self.do_original_locs = do_original_locs
        if do_log1p and do_sqrt:
            ValueError('Should do either log1p or sqrt for preprocess')
        self.do_log1p = do_log1p
        self.do_sqrt = do_sqrt
        self.n_comp = n_comp
        self.pX = None
        self.do_hvgs = do_hvgs
        self.n_hvgs = n_hvgs
        self.name = name
        self.outdir = outdir
        
        if self.outdir is not None:
            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)

        # preprocess
        if self.do_hvgs:
            self.hvgs, self.ihvgs = self.get_hvgs(n_hvgs=self.n_hvgs) # computing hvgs one on full data
        self.pX, self.lX, self.pca = self.preprocess(self.X, return_pca=True)
        
        self.dim = None
        self.P = None 
        if D is None:
            D, P = T.ds.get_pairwise_distances(self.pX, return_predecessors=True)
            self.P = P

        self.D = D
        self.milestone_network = milestone_network
        self.meta = meta if meta is not None else pd.DataFrame(index=cellnames)
        self.meta['Trajectory_idx'] = np.arange(self.ncells)
        
        # for expression analysis
        self.exp_corr_hvgs = None
        self.n_buckets = None
        self.buckets_mean = None

    
    def preprocess(self, X, verbose=False, return_pca=False):
        """
        Standard preprocess
        Optional count transformation (log1p or sqrt),
        followed dimensionality reduction (highly variable genes or PCA)
        :param X: expression counts (cells x genes)
        :param return_pca: return PCA object
        :return: 
        pX - transformed and reduced expression
        lX - transformed expression
        """
        # return data as is without preprocessing
        pX = X.copy()
        lX = X.copy()
        pca = None
        do_preprocess = self.do_preprocess

        # use the original cell locations (similar to without pp but can be a latent representation)
        if (self.do_original_locs) and (self.pX is not None):
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
            if self.do_hvgs:
                if verbose:
                    print('hvgs representation')
                pX = lX.loc[:, self.hvgs]
            else:
                # pca computation
                pca = PCA(n_components=self.n_comp)
                pX = pca.fit_transform(lX)
                pcnames = ['PC%d' % (i+1) for i in np.arange(pX.shape[1])]
                pX = pd.DataFrame(pX, index=X.index, columns=pcnames)
        
        if return_pca:
            return pX, lX, pca

        return pX, lX


    def get_hvgs(self, n_hvgs=1000, perc_top_hvgs=None, **kwargs):
        """
        Uses Scanpy highly_variable_genes computation
        :return:
        """
        n_hvgs = int(self.ngenes * perc_top_hvgs) if perc_top_hvgs is not None else n_hvgs
        adata = sc.AnnData(self.X)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, **kwargs)
        adata.var['genename'] = self.X.columns
        ihvgs = np.where(adata.var['highly_variable'])[0]
        hvgs = list(adata.var.iloc[ihvgs]['genename'])
        
        return hvgs, ihvgs


    def subsample_counts(self, pc, pt):
        """
        Subsample cells and reads
        :param X: expression counts
        :param pc: cell capture probability
        :param pt: transcript capture probability
        :return:
            subsampled expression
            index of subsampled cells
        """
        n = int(self.ncells * pc)
        if n < self.ncells:
            ix = list(np.random.choice(self.cellnames, n, replace=False))
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
        :param pc: cell capture probability
        :param pt: transcript capture probability
        :param ix: index of cells to subsample
        :return:sX, psX, lsX, psD, sD, psP, ix, pca
        sX - subsampled expression
        psX - subsampled reduced expression
        lsX - subsampled transformed expression
        psD - subsampled reduced distances
        sD - subsampled original distances
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
        psD, psP = T.ds.get_pairwise_distances(psX, return_predecessors=True, verbose=verbose) 
        
        return sX, psX, lsX, psD, sD, psP, ix, pca



    def downsample_params(self, B, Pc=None, Pt=None, min_reads=20, verbose=False):
        """
        Filtering downsampling params
        :param B: sequencing budget
        :param Pc: cell downsample probabilities
        :param Pt: read downsample probabilities
        :param min_reads: minimum number of reads per cell on average
        :return:
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


    def evaluate(self, sX, psX, lsX, psD, sD, psP, ix, pca, pc, pt, 
                comp_pseudo_corr=False, pseudo_use='dpt',
                comp_exp_corr=False, verbose=False, plot=False,):
        """
        Computes statistics of downsampled data
        :param sX: sampled expression
        :param psX: latent representation of sampled expression
        :param psD: distances of latent representation of sampled expression
        :param sD: original distances of sampled data
        :param psP: 
        :param ix: index of sampled cells
        :param comp_deltas: whether to compute side of each PC (Delta)
        :param comp_nn_dist: whether to compute nearest neighbor distances
        :param comp_pseudo_corr: whether to compute pseudotime correlation
        :param comp_exp_corr: 
        :param comp_vertex_length:
        :param comp_covariance: compare covariance of gene expression space
        :param comp_covariance_latent: compare covariance of latent space
        :param comp_pc_err: compare PC error
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
            try:
                present_groups = [g for g in self.group_order if g in smeta[self.group_col].unique()]
                source = present_groups[0]
                sinks = present_groups[-1]
                a, _ = self.eval_linear_regression(X=lsX, meta=smeta, group_col=self.group_col, source=source, sinks=sinks)
                pseudo = -a[source]

                # pseudo = T.dw.get_pseudo(sX, smeta, pX=psX.values, plot=plot)

                if plot:
                    plt.scatter(psX.values[:,0], psX.values[:,1], c=pseudo)
                

                dpt_corr = np.corrcoef(smeta[pseudo_use], pseudo)[0, 1]
                report['dpt_corr'] = dpt_corr
            except:
                pseudo = None
                print('Could not compute pseudotime')
 
        if comp_exp_corr and (pseudo is not None):
            or_s_buckets_mean = T.dw.get_mean_bucket_exp(sX[self.exp_corr_hvgs], smeta[pseudo_use], n_buckets=self.n_buckets, plot=plot)
            s_buckets_mean = T.dw.get_mean_bucket_exp(sX[self.exp_corr_hvgs], pseudo, n_buckets=self.n_buckets, plot=plot)
            or_exp_corr = T.dw.expression_correlation(self.buckets_mean, or_s_buckets_mean)
            exp_corr = T.dw.expression_correlation(self.buckets_mean, s_buckets_mean)
            report['exp_corr'] = exp_corr
            report['or_exp_corr'] = or_exp_corr

        return report


    def eval_linear_regression(self, group_col, source, sinks, use_rep='log1p', X=None, meta=None):
        """
        Mean norm to subspace of source and sink cell types (Pusuluri et al. 2019)
        """

        # g - number of genes
        # n - number of cells
        # p - number of types

        # get mean expression of source and sink cell types
        X = (self.lX if use_rep == 'log1p' else self.X) if X is None else X
        meta = self.meta if meta is None else meta
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

        return a, S_perp_norm

    def compute_tradeoff(self, B, Pc=None, Pt=None, repeats=50, verbose=False, plot=False,
                        comp_pseudo_corr=False, pseudo_use='dpt', 
                        comp_exp_corr=False, 
                        hvgs=None, n_buckets=5, **kwargs):
        """
        Compute reconstruction error for subsampled data within budget opt
        :param X: counts data
        :param D_true: geodesic distances
        :param B: counts budget
        :param Pc: cell capture probabilities
        :param repeats: number of repeats
        :param verbose: print messages
        :param plot: plot reduced expression for each set of downsample params
        :return:
            dataframe with sampling params and errors
        """

        dws_params = self.downsample_params(B, Pc, Pt, verbose)

        # pre-compute for entire data
        if comp_pseudo_corr or comp_exp_corr:
            if pseudo_use == 'dpt':

                # self.meta[pseudo_use] = T.dw.get_pseudo(self.X, self.meta, pX=self.pX.values, plot=plot)
                
                source = self.group_order[0]
                sinks = self.group_order[-1]
                a, _ = self.eval_linear_regression(group_col=self.group_col, source=source, sinks=sinks)
                self.meta[pseudo_use] = -a[source]
                
                if plot:
                    plt.scatter(self.pX.values[:,0], self.pX.values[:,1], c=self.meta[pseudo_use])

        if comp_exp_corr:
            # select genes for reconstruction evaluation
            if self.exp_corr_hvgs is None:

                hvgs = self.get_hvgs(perc_top_hvgs=0.10)[0]

                # select genes with high expression
                hvgs_mean = self.X[hvgs].mean()
                n_hhvgs = min(50, len(hvgs))
                self.exp_corr_hvgs = list(hvgs_mean.sort_values()[-n_hhvgs:].index)

                if verbose:
                    print('Using %d genes' % len(self.exp_corr_hvgs))
            self.n_buckets = n_buckets
            self.buckets_mean = T.dw.get_mean_bucket_exp(self.X[self.exp_corr_hvgs], self.meta[pseudo_use], n_buckets=self.n_buckets, plot=plot)

        # evaluate subsampled data
        L = []
        for k in range(repeats):
            if verbose:
                print(k)
            for _, row in dws_params.iterrows():

                pc = row['pc']
                pt = row['pt']

                # sample
                try:
                    subsample_result = self.subsample(pc, pt, verbose=verbose)
                except np.linalg.LinAlgError as err:
                    if verbose:
                        print(f'When downsampling with cell probability {pc} and read probability {pt}, got LinAlgError.')
                    continue
                
                report = self.evaluate( *subsample_result, pc=pc, pt=pt, 
                                        comp_pseudo_corr=comp_pseudo_corr, pseudo_use=pseudo_use,
                                        comp_exp_corr=comp_exp_corr, verbose=verbose, plot=plot, **kwargs)
                         
                report = {'pc': pc, 'pt': pt, 'B': B, 
                          'log pc': np.log(pc), 'log pt': np.log(pt),  
                          **report}
                L.append(report)

        L = pd.DataFrame(L)
        return L

