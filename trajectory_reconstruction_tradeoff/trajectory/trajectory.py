import os
from tabnanny import verbose
import skdim
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import trajectory_reconstruction_tradeoff as T
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.linalg import svd

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
    do_preprocess=True, do_log1p=True,  do_sqrt=False, do_original_locs=False, n_comp=10, do_hvgs=False, n_hvgs=100,
    by_radius=False, radius=None, radius_fold=None, name=''):
        """Initialize the tissue using the counts matrix and, if available, ground truth distance matrix.
        X      -- counts matrix (cells x genes)
        D      -- if available, ground truth cell-to-cell distances (cells x cells)
        meta   -- other available information per cell
        outdir  -- folder path to save the plots and data
        do_preprocess -- if False skips complete preprocessing step
        do_log1p -- if to perform log1p transformation transformation
        do_sqrt -- if to perform sqrt transformation transformation
        do_original_locs -- if to use original("true") cell locations
        radius -- if provided, use only cells within this radius ball
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
                if D is not None:
                    D = D[idx, :][:, idx]

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
        self.radius = radius
        self.by_radius = by_radius
        self.radius_fold = radius_fold
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
        # self.dim =  self.get_dimension() 
        
        # if self.pca:
            # self.projection = self.compute_projection(self.pca)
        # self.projection = self.compute_projection(self.lX)
        
        self.P = None 

        if D is None:
            D,P = T.ds.get_pairwise_distances(self.pX.values, return_predecessors=True, 
            by_radius=self.by_radius, radius=self.radius, radius_fold=self.radius_fold, dim=self.dim)
            self.P = P # predecessors
        # D = D / np.max(D) # TODO: removed this late! 

        self.V = None # heavy to compute so only if necessary
        self.D = D
        self.milestone_network = milestone_network
        self.meta = meta if meta is not None else pd.DataFrame(index=cellnames)
        self.meta['Trajectory_idx'] = np.arange(self.ncells)
        
        # for arias-castro analysis
        self.density_0 = None
        self.reach_0 = None

        # for expression analysis
        self.exp_corr_hvgs = None
        self.n_buckets = None
        self.buckets_mean = None

    # def set_n_comp(self, n_comp):
    #     """
    #     Edit number of components
    #     :param n_comp:
    #     """
    #     self.n_comp = n_comp


    # def set_log1p(self, do_log1p):
    #     """
    #     Set whether to apply log1p
    #     :param do_log1p:
    #     """
    #     self.do_log1p = do_log1p

    # @staticmethod #TODO: make static?
    def preprocess(self, X, verbose=False, return_pca=False):
        """
        Standard preprocess
        Optional log1p or sqrt,
        followed by (optional) highly variable genes or PCA reduction
        :param X: expression counts (cells x genes)
        :param verbose: print progress
        :param return_pca: return PCA object
        :return: preprocessed expression
        """
        # return data as is without preprocessing
        pX = X.copy()
        lX = X.copy()
        pca = None
        do_preprocess = self.do_preprocess

        # use the original cell locations (similar to without pp but can be a latent representation)
        if (self.do_original_locs) and (self.pX is not None):
            # return self.pX.loc[X.index]
            lX = self.lX.loc[X.index]
            pX = self.pX.loc[X.index]
            pca = self.pca
            do_preprocess = False

        if do_preprocess:
            # collapsing operation, log1p, sqrt
            lX = X.copy()
            if self.do_log1p:
                if verbose:
                    print('do_log1p')
                lX = np.log1p(X)
            elif self.do_sqrt:
                if verbose:
                    print('do_sqrt')
                lX = np.sqrt(X)
            if self.do_hvgs:
                if verbose:
                    print('hvgs representation')
                pX = lX.loc[:, self.hvgs]
            else:
                # pca computation
                pca = PCA(n_components=self.n_comp, svd_solver='full')
                pX = pca.fit_transform(lX)
                pcnames = ['PC%d' % (i+1) for i in np.arange(pX.shape[1])]
                pX = pd.DataFrame(pX, index=X.index, columns=pcnames)
        
        if return_pca:
            return pX, lX, pca

        return pX, lX

    def compute_projection(self, lX, by_hvgs=True):
        """
        Computing projection matrix (over hvgs)
        lX - (cells x genes)
        """
        X = lX.T

        # center data
        X_cen = (X.T - X.mean(axis=1)).T  # substract column means
        # X_cen = X - X.mean(axis=0)
        # compute svd decomposition of X
        U, _, _ = svd(X_cen, full_matrices=False) # U is (genes x min(genes,cells))

        # compute projection matrix
        if by_hvgs:
            U = U[self.ihvgs] # only over hvgs
        projection = U @ U.T

        return projection


    # def compute_projection(self, pca, by_hvgs=True):
    #     """
    #     Computing projection matrix (over hvgs)
    #     """
    #     U = pca.components_.T # I think should compute over the transpose but I think this is alright (and much cheaper in memory)
    #     if by_hvgs:
    #         U = U[self.ihvgs]
    #     projection = U @ U.T
    #     return projection

    # @staticmethod # TODO: make static?
    def get_dimension(self, verbose=False):
        """
        Get dimension of latent representation
        """
        #estimate global intrinsic dimension
        # danco = skdim.id.DANCo().fit(self.pX)
        #estimate local intrinsic dimension (dimension in k-nearest-neighborhoods around each point):
        lpca = skdim.id.lPCA().fit_pw(self.pX,
                                    n_neighbors=100,
                                    n_jobs=1)

        #get estimated intrinsic dimension
        if verbose:
            # print(f'DANCo dimension estimate: {danco.dimension_}')
            print(f'Mean lpca dimension estimates are: {np.mean(lpca.dimension_pw_)}')
        return np.mean(lpca.dimension_pw_)

    # @staticmethod # TODO: make static?
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
        ix = np.random.choice(self.ncells, n, replace=False)
        sX = self.X.iloc[ix, :]
        
        cellnames = sX.index
        genenames = sX.columns
        if pt < 1:
            sX = sX.astype(int)
            sX = np.random.binomial(sX, pt)
        sX = pd.DataFrame(sX, index=cellnames, columns=genenames)
        return sX, ix

    def subsample(self, pc, pt, ix=None, verbose=False):
        """
        Subsample cells and reads
        :param X: expression counts
        :param D_true: distances
        :param pc: cell capture probability
        :param pt: transcript capture probability
        :param ix: index of cells to subsample
        :param n_pc: number of PCs for reduced expression
        :return:
            subsampled expression
            subsampled reduced expression
            distances (subsampled expression)
            distances (subsampled original distances)
            index of subsampled cells
        """
        if ix is None:
            sX, ix = self.subsample_counts(pc, pt)
        else:
            sX = self.X.iloc[ix, :]
        sD = self.D[ix][:, ix]  # subsampled ground truth pairwise distances
        # sD_max = np.max(sD) #TODO: BIG CHANGE
        # sD = sD / sD_max

        psX, lsX, pca = self.preprocess(sX, return_pca=True)
        psD, psP = T.ds.get_pairwise_distances(psX.values, return_predecessors=True,
                                               by_radius=self.by_radius, radius=self.radius, radius_fold=self.radius_fold, dim=self.dim, verbose=verbose) #TODO: BIG CHANGE , psD_ma, radius=self.radiusx
        
        return sX, psX, lsX, psD, sD, psP, ix, pca



    def _downsample_params(self, B, Pc=None, Pt=None, verbose=False):
        """
        Filtering downsampling params
        :param B: sequencing budget
        :param Pc: cell downsample probabilities
        :param Pt: read downsample probabilities
        :return:
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
            if verbose: print('Restricting Pc to range of available cells/PC dimensions')
            dws_params = dws_params[~cond]

        cond = dws_params['pc'] > 0.90
        if np.any(cond):
            if verbose: print('Restricting Pc so can subsample')
            dws_params = dws_params[~cond]

        cond = dws_params['pc'] < B
        if np.any(cond):
            if verbose: print('Restricting Pc to budget limit')
            dws_params = dws_params[~cond]

        cond = dws_params['pt'] > 1
        if np.any(cond):
            if verbose: print('Restricting Pt to 1')
            dws_params = dws_params[~cond]

        cond = dws_params['pt'] < epsilon
        if np.any(cond):
            if verbose: print('Pt may be too low')

        if dws_params.shape[0] == 0:
            ValueError('All params were filtered out')

        return dws_params


    def evaluate(self, sX, psX, lsX, psD, sD, psP, ix, pca, pc, pt, 
                comp_deltas=False, comp_nn_dist=True, 
                comp_pseudo_corr=False, pseudo_use='dpt',
                comp_exp_corr=False, 
                comp_vertex_length=False, 
                comp_covariance=False, 
                comp_covariance_latent=False, 
                comp_pc_err=False, 
                comp_reach=False, comp_density=False, 
                comp_proj_err=False, verbose=False, plot=False,):
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
        
        smeta = self.meta.iloc[ix].copy()

        dmax_psD = np.max(psD); npsD = psD / dmax_psD
        dmax_sD = np.max(sD); nsD = sD / dmax_sD

        # compute error
        l1, l2, ldist, fcontrac, fexpand, lsp = T.ds.compare_distances(nsD, npsD)

        
        report = {'nc': nc, 'nr': nr, 'Br': sX.sum().sum(),
                  'l1': l1, 'l2': l2, 'ldist': ldist, 'fcontrac': fcontrac, 'fexpand': fexpand, 'lsp': lsp, 
                  'dmax_psD': dmax_psD, 'dmax_sD': dmax_sD}

        if comp_deltas:
            Delta_vals = psX.max(0) - psX.min(0)
            Deltas = {'Delta%d' % i: Delta_vals[i] for i in np.arange(psX.shape[1])}
            report = {**report, **Deltas}
            report['Delta'] = pdist(psX).max()

        if comp_nn_dist:
            idxmin = (sD + sD.max() * np.eye(nc)).argmin(0) # get index of nearest neighbors
            report['nn_dist_sD'] = sD[np.arange(nc), idxmin].mean()
            report['nn_dist_nsD'] = nsD[np.arange(nc), idxmin].mean()

            report['nn_diff_n'] = np.abs((nsD - npsD))[np.arange(nc), idxmin].mean()

            idxmin = (psD + psD.max() * np.eye(nc)).argmin(0)
            report['nn_dist_psD'] = psD[np.arange(nc), idxmin].mean()
            report['nn_dist_npsD'] = npsD[np.arange(nc), idxmin].mean()

        if comp_pseudo_corr or comp_exp_corr:
            try:
                # pseudo = T.dw.get_pseudo(sX, smeta, pX=psX, plot=plot)
                
                present_groups = [g for g in self.group_order if g in smeta[self.group_col].unique()]
                source = present_groups[0]
                sinks = present_groups[-1]
                a, _ = self.eval_linear_regression(X=lsX, meta=smeta, group_col=self.group_col, source=source, sinks=sinks)
                pseudo = -a[source]

                if plot:
                    plt.scatter(psX.values[:,0], psX.values[:,1], c=pseudo)
                

                dpt_corr = np.corrcoef(smeta[pseudo_use], pseudo)[0, 1]
                # print(dpt_corr)
                report['dpt_corr'] = dpt_corr
            except:
                pseudo = None
                print('Could not compute pseudotime')

        if comp_vertex_length:
            psV = T.ds.compute_path_vertex_length(psP)
            ratio_n_vertices = (psV / nc) / (self.V[ix][:, ix] / self.ncells) # handling disconnections?
            ratio_n_vertices[range(nc), range(nc)] = 0
            weights = np.ones((nc, nc)) - np.eye(nc)
            avg_ratio_n_vertices = np.average(ratio_n_vertices, weights=weights) 
            
            # import matplotlib.pyplot as plt
            # plt.clf(); plt.hist((self.V[ix][:, ix]).flatten(), alpha=0.5); 
            # plt.hist(psV.flatten(), alpha=0.5); 
            # plt.title(f'{self.name}, avg ratio number vertices: {avg_ratio_n_vertices}')
            # plt.savefig(f'dummy_{self.name}.png')
            
            report['avg_ratio_n_vertices'] = avg_ratio_n_vertices
        
        if comp_covariance:
            sC = T.ds.compute_covariance(sX)
            report['cov_err'] = np.linalg.norm(self.C - sC)

        if comp_covariance_latent:
            psC = T.ds.compute_covariance(psX)
            report['cov_latent_err'] = np.linalg.norm(self.pC - psC)
        
        if comp_exp_corr and (pseudo is not None):
            or_s_buckets_mean = T.dw.get_mean_bucket_exp(sX[self.exp_corr_hvgs], smeta[pseudo_use], n_buckets=self.n_buckets, plot=plot)
            s_buckets_mean = T.dw.get_mean_bucket_exp(sX[self.exp_corr_hvgs], pseudo, n_buckets=self.n_buckets, plot=plot)
            or_exp_corr = T.dw.expression_correlation(self.buckets_mean, or_s_buckets_mean)
            exp_corr = T.dw.expression_correlation(self.buckets_mean, s_buckets_mean)
            report['exp_corr'] = exp_corr
            report['or_exp_corr'] = or_exp_corr

        # compute change of pc direction
        if comp_pc_err:
            # max_err = np.sqrt(2) # heimberg says they normalize by sqrt(2) but I don't think that's correct
            pc_err = np.linalg.norm(pca.components_ - self.pca.components_, axis=1) #/ max_err
            for pc_dim in np.arange(self.n_comp):
                report[f'PC{pc_dim+1}_err'] = pc_err[pc_dim]

        if comp_reach and pca:
            sX_all, _ = self.subsample_counts(pc=1, pt=pt) # not ideal bc subsample can change
            _, lsX_all = self.preprocess(sX_all)
            if verbose:
                print('Computing reachability is appropriate only with all/most cells included')
            psX_all = pca.transform(lsX_all) # applying pca on all cells
            reach = T.ds.compute_reach(psX_all)
            report['reach'] = reach
            # if pca:
            #     pX_all = pca.transform(X) #TODO: should be lX, right now appropriate only when there is no log1p
            #     reach_org_proj = T.ds.compute_reach(pX_all)
            #     report['reach_org_proj'] = reach_org_proj

        if comp_density:
            # density_0 = T.ds.compute_density(sD)
            density = T.ds.compute_density(sD)
            pdensity = T.ds.compute_density(psD)
            density_est = (np.log(nc) / nc) ** (1 / self.dim)
            # report['density_0'] = density_0
            report['density'] = density
            report['pdensity'] = pdensity
            report['density_est'] = density_est

        if comp_proj_err:
            if verbose:
                print('Assuming no log1p transform')
            # projection = self.compute_projection(pca) # need left eigenvectors
            projection = self.compute_projection(lsX) # need left eigenvectors
            report['proj_err'] = np.linalg.norm(self.projection - projection, ord=2)

        return report


    def eval_dimensionality_perp(self, group_col, source, sinks, use_rep='log1p'):
        """
        Mean norm to subspace of source and sink cell types (Pusuluri et al. 2019)
        """

        # g - number of genes
        # n - number of cells
        # p - number of types

        # get mean expression of source and sink cell types
        X = self.lX if use_rep == 'log1p' else self.X
        group_X = pd.concat((self.meta[group_col], X), axis=1)
        group_mean = group_X.groupby(group_col).mean()
        sinks = sinks if isinstance(sinks, list) else [sinks]
        source_exp = group_mean.loc[source]
        sinks_exp = group_mean.loc[sinks]

        S = X.T # g x n - expression
        g = S.shape[0]

        # xsi = adata.X[[1,-1],:].T # g x p - cell type expression
        # grp_ids = [source] + fates
        xsi = np.vstack((source_exp, sinks_exp)).T # g x p - cell type mean expression

        m = 1.0/g * np.dot(S.T, xsi) # n x p - similarity of expression to cell type expression

        A = 1.0/g * np.dot(xsi.T, xsi) # p x p - similarity bw types
        Ainv = np.linalg.pinv(A) # p x p - inverse A

        a = np.dot(m, Ainv) # n x p - weighted similarity, 

        S_proj = np.dot(xsi, a.T) # g x n - projection of expression on type ab 
        S_perp = (S - S_proj) # remaining expression

        S_perp_norm = np.linalg.norm(S_perp, axis=0)

        # return np.mean(S_perp_norm)
        a = pd.DataFrame(a, index=self.meta.index, columns=[source] + sinks)
        S_perp_norm = pd.Series(S_perp_norm, index=self.meta.index)

        return a, S_perp_norm



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

        # return np.mean(S_perp_norm)
        a = pd.DataFrame(wT, index=meta.index, columns=[source] + sinks)
        S_perp_norm = pd.Series(S_perp_norm, index=meta.index)

        return a, S_perp_norm

    def compute_tradeoff(self, B, Pc=None, Pt=None, repeats=50, verbose=False, plot=False,
                        comp_pseudo_corr=False, pseudo_use='dpt', 
                        comp_exp_corr=False, 
                        comp_vertex_length=False, 
                        comp_covariance=False, 
                        comp_covariance_latent=False, 
                        comp_reach=False, comp_density=False, 
                        comp_proj_err=False,
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

        dws_params = self._downsample_params(B, Pc, Pt, verbose)

        if comp_vertex_length and self.V is None:
            if self.P is None:
                print('Distance matrix provided. Need to edit predecessor computation according to D provided.')
                _,P = T.ds.get_pairwise_distances(self.pX.values, return_predecessors=True, 
                by_radius=self.by_radius, radius=self.radius, radius_fold=self.radius_fold, dim=self.dim)

            self.V = T.ds.compute_path_vertex_length(self.P)

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

        if comp_covariance:
            self.C = T.ds.compute_covariance(self.X)

        if comp_covariance_latent:
            self.pC = T.ds.compute_covariance(self.pX)

        if comp_proj_err:
            if self.projection is None:
                self.projection = self.compute_projection(self.lX)

        if comp_density:
            if self.density_0 is None:
                self.density_0 = T.ds.compute_density(self.D)

        if comp_reach:
            if self.reach_0 is None:
                self.reach_0 = T.ds.compute_reach(self.pX) #TODO: give dimension as input            

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
                                        comp_exp_corr=comp_exp_corr, 
                                        comp_vertex_length=comp_vertex_length, 
                                        comp_covariance=comp_covariance, 
                                        comp_covariance_latent=comp_covariance_latent, 
                                        comp_reach=comp_reach, comp_density=comp_density, 
                                        comp_proj_err=comp_proj_err, verbose=verbose, plot=plot, **kwargs)
                         
                report = {'pc': pc, 'pt': pt, 'B': B, 
                          'log pc': np.log(pc), 'log pt': np.log(pt), 
                          'sqrt inv pc': np.sqrt(1/pc), 'sqrt inv pt': np.sqrt(1/pt), 
                          **report}
                L.append(report)

                # if plot and (k == 0):
                #     tit = 'B = %.5f, pc = %.5f, pt = %.5f \n l1 = %.2f, l2 = %.2f' % (B, pc, pt, l1, l2)
                #     T.pl.plot_pca2d(psX, title=tit)

        L = pd.DataFrame(L)
        return L



if __name__ == '__main__':
    # size_factor = 5
    # T.io.set_size_factor(size_factor=size_factor)
    ncells = 1000
    nsegs = 5
    nper_seg = int(ncells / nsegs)
    newick = "((C:%d)B:%d,(D:%d)E:%d)A:%d;" % ((nper_seg,) * nsegs)
    # newick = '((((A:%d)B:%d)C:%d)D:%d)E:%d;' % ((nper_seg,) * nsegs)
    from scipy.spatial.distance import cdist
    # X, D_true, meta = T.io.simulate(newick)
    datasets = ['beta', 'hayashi']
    dataset = 'linear_rep0'
    # for dataset in datasets:
        # print(dataset)
    X, _, meta, mn = T.io.read_dataset(dataset=dataset, dirname='datasets/')
    traj = Trajectory(X, meta=meta, milestone_network=mn)
    L = traj.compute_tradeoff(B=0.001, Pc=[0.84], comp_pseudo_corr=True, comp_exp_corr=True, repeats=1, plot=False, pseudo_use='Trajectory_idx' )
    print(L)
    # source = mn.iloc[0]['from']
    # after_source = mn.iloc[0]['to']
    # sink = mn.iloc[-1]['to']
    # source_exp = traj.X[traj.meta['milestone_id'] == source].mean()
    # aftersource_exp = traj.X[traj.meta['milestone_id'] == after_source].mean()
    # sink_exp = traj.X[traj.meta['milestone_id'] == sink].mean()
    # print(np.linalg.norm(source_exp-sink_exp) / traj.ngenes)
        
        # check expression distance bw first and last timepoint

#     X = pd.DataFrame(X)

#     adata = sc.AnnData(X)
#     adata.obs = meta.loc[X.index]
#     sc.pp.log1p(adata)
#     sc.tl.pca(adata)  # recomputing since 10 pcs give disconnected pseudotime
#     sc.pl.pca(adata)
# #     sc.pp.neighbors(adata, method='gauss', use_rep='X_pca')
# #     sc.tl.diffmap(adata)
# #     adata.uns['iroot'] = 0  # np.where(adata.obs_names == adata.obs[idx_col].idxmin())[0][0]
# #     sc.tl.dpt(adata)
# #     sc.pl.pca(adata, color=['pseudotime', 'dpt_pseudotime'])
# #     print(np.corrcoef(adata.obs['pseudotime'], adata.obs['dpt_pseudotime'])[0][1])
# #
#     # dirname = '/Users/nomo/PycharmProjects/Tree_Reconstruct_Limitations/datasets/'
#     # dataset = 'prosstt_sf%d' % size_factor
#     # X.to_csv(os.path.join(dirname, 'counts_%s.csv' % dataset))
# #
#     # meta.to_csv(os.path.join(dirname, '%s_cell_info.csv' % dataset))
# #
#     # D_true = pd.DataFrame(D_true, columns=X.index, index=X.index)
#     # D_true.to_csv(os.path.join(dirname, 'geodesic_%s.csv' % dataset))
    
# #
# #     # traj = Trajectory(X, D_true)
#     traj = Trajectory(X, D_true, meta=meta)
#     traj.compute_tradeoff(0.002, [0.3])
#     print('done')
