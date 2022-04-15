import os
import numpy as np
import pandas as pd
import scanpy as sc
import trajectory_reconstruction_tradeoff as T
# from trajectory_reconstruction_tradeoff import plotting as P
from sklearn.decomposition import PCA
# from trajectory_reconstruction_tradeoff.distances.compute_dists import get_pairwise_distances
epsilon = 10e-10


class Trajectory():
    """
    Trajectory object
    """

    def __init__(self, X, D=None, meta=None, outdir=None, do_preprocess=True, do_log1p=True,  do_sqrt=False, n_comp=10):
        """Initialize the tissue using the counts matrix and, if available, ground truth distance matrix.
        X      -- counts matrix (cells x genes)
        D      -- if available, ground truth cell-to-cell distances (cells x cells)
        meta   -- other available information per cell
        outdir  -- folder path to save the plots and data"""

        self.ncells, self.ngenes = X.shape
        if isinstance(X, pd.DataFrame) and meta is not None:
            if (X.index != meta.index).any():
                print('Counts and metadata index differ')
                return
        if not isinstance(X, pd.DataFrame):
            genenames = ['g%d' % i for i in range(self.ngenes)]
            cellnames = ['c%d' % i for i in range(self.ncells)]
            X = pd.DataFrame(X, columns=genenames, index=cellnames)
        self.X = X
        self.do_preprocess = do_preprocess
        if do_log1p and do_sqrt:
            ValueError('Should do either log1p or sqrt for preprocess')
        self.do_log1p = do_log1p
        self.do_sqrt = do_sqrt
        self.n_comp = n_comp
        self.pX = self.preprocess(self.X)

        if D is None:
            D = T.ds.get_pairwise_distances(self.pX)[0]
        D = D / np.max(D)

        self.D = D
        self.meta = meta if meta is not None else pd.DataFrame(index=cellnames)
        self.meta['original_idx'] = np.arange(self.ncells)
        self.outdir = outdir


    def set_n_comp(self, n_comp):
        """
        Edit number of components
        :param n_comp:
        """
        self.n_comp = n_comp


    def set_log1p(self, do_log1p):
        """
        Set whether to apply log1p
        :param do_log1p:
        """
        self.do_log1p = do_log1p


    def preprocess(self, X, verbose=False):
        """
        Standard preprocess
        """
        if not self.do_preprocess:
            if verbose:
                print('no preprocess')
            return X.copy()
        pca = PCA(n_components=self.n_comp, svd_solver='full')
        lX = X.copy()
        if self.do_log1p:
            if verbose:
                print('do_log1p')
            lX = np.log1p(X)
        if self.do_sqrt:
            if verbose:
                print('do_sqrt')
            lX = np.sqrt(X)
        pX = pca.fit_transform(lX)
        return pX

    def get_hvgs(self, n_hvgs=1000, **kwargs):
        """
        Uses Scanpy highly_variable_genes computation
        :return:
        """
        adata = sc.AnnData(self.X)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, **kwargs)
        adata.var['genename'] = self.X.columns
        hvgs = list(adata.var[adata.var['highly_variable']]['genename'])
        return hvgs

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
        sX = sX.astype(int)
        cellnames = sX.index
        genenames = sX.columns
        if pt < 1:
            sX = np.random.binomial(sX, pt)
        sX = pd.DataFrame(sX, index=cellnames, columns=genenames)
        return sX, ix

    def subsample(self, pc, pt):
        """
        Subsample cells and reads
        :param X: expression counts
        :param D_true: distances
        :param pc: cell capture probability
        :param pt: transcript capture probability
        :param n_pc: number of PCs for reduced expression
        :return:
            subsampled expression
            subsampled reduced expression
            distances (subsampled expression)
            distances (subsampled original distances)
            index of subsampled cells
        """
        sX, ix = self.subsample_counts(pc, pt)
        sD = self.D[ix][:, ix]  # subsampled ground truth pairwise distances
        sD = sD / np.max(sD)

        psX = self.preprocess(sX)
        psD = T.ds.get_pairwise_distances(psX)[0]
        return sX, psX, psD, sD, ix

    def _downsample_params(self, B, Pc=None, Pt=None, verbose=False):
        """
        Filtering downsampling params
        :param B:
        :param Pc:
        :param Pt:
        :return:
        """
        if (Pc is None) and (Pt is None):
            ValueError('Need to set Pc or Pt')
        if Pt is None:
            Pt = [B / pc for pc in Pc]
        if Pc is None:
            Pc = [B / pt for pt in Pt]

        dws_params = pd.DataFrame({'pc': Pc, 'pt': Pt})

        if B > 1:
            ValueError('B needs to be smaller than 1')

        cond = dws_params['pc'] < 1 / self.ncells
        if np.any(cond):
            if verbose: print('Restricting Pc to range of available cells')
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

    def compute_tradeoff(self, B, Pc=None, Pt=None, repeats=50, verbose=False, plot=False,
                         comp_pseudo_corr=False, comp_exp_corr=False, comp_deltas=False,
                         hvgs=None, n_buckets=10):
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

        if comp_pseudo_corr or comp_exp_corr:
            self.meta['dpt'] = T.dw.get_pseudo(self.X, self.meta, pX=self.pX)

        if comp_exp_corr:
            # select genes for reconstruction evaluation
            if hvgs is None:
                perc_top_hvgs = 0.10
                n_hvgs = int(self.ngenes * perc_top_hvgs)  # 10 top hvgs
                hvgs = self.get_hvgs(n_hvgs=n_hvgs)
                hvgs_mean = self.X[hvgs].mean()
                n_hhvgs = min(20, len(hvgs))
                hvgs = list(hvgs_mean.sort_values()[-n_hhvgs:].index)
                if verbose:
                    print('Using %d genes' % len(hvgs))

            bucket_mean = T.dw.get_mean_bucket_exp(self.X[hvgs], self.meta['dpt'], n_buckets=n_buckets)

        L = []

        for k in range(repeats):
            if verbose:
                print(k)
            for _, row in dws_params.iterrows():

                pc = row['pc']
                pt = row['pt']

                # sample
                sX, psX, psD, sD, ix = self.subsample(pc, pt)
                smeta = self.meta.iloc[ix].copy()

                # compute error
                l1, l2, l3, lsp = T.ds.compare_distances(psD, sD)

                report = {'pc': pc, 'pt': pt, 'B': B,
                          'l1': l1, 'l2': l2, 'l3': l3, 'lsp': lsp}

                if comp_deltas:
                    Delta_vals = psX.max(0) - psX.min(0)
                    Deltas = {'Delta%d' % i: Delta_vals[i] for i in np.arange(psX.shape[1])}
                    report = {**report, **Deltas}

                if comp_pseudo_corr or comp_exp_corr:
                    pseudo = T.dw.get_pseudo(sX, smeta, pX=psX)
                    dpt_corr = np.corrcoef(smeta['dpt'], pseudo)[0, 1]
                    report['dpt_corr'] = dpt_corr

                if comp_exp_corr:
                    or_s_bucket_mean = T.dw.get_mean_bucket_exp(sX[hvgs], smeta['dpt'], n_buckets=n_buckets)
                    s_bucket_mean = T.dw.get_mean_bucket_exp(sX[hvgs], pseudo, n_buckets=n_buckets)
                    or_exp_corr = T.dw.expression_correlation(bucket_mean, or_s_bucket_mean)
                    exp_corr = T.dw.expression_correlation(bucket_mean, s_bucket_mean)
                    report['exp_corr'] = exp_corr
                    report['or_exp_corr'] = or_exp_corr


                L.append(report)

                if plot and (k == 0):
                    tit = 'B = %.5f, pc = %.5f, pt = %.5f \n l1 = %.2f, l2 = %.2f' % (B, pc, pt, l1, l2)
                    T.pl.plot_pca2d(psX, title=tit)

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

    X, D_true, meta = T.io.simulate(newick)
    X = pd.DataFrame(X)

    adata = sc.AnnData(X)
    adata.obs = meta.loc[X.index]
    sc.pp.log1p(adata)
    sc.tl.pca(adata)  # recomputing since 10 pcs give disconnected pseudotime
    sc.pl.pca(adata)
#     sc.pp.neighbors(adata, method='gauss', use_rep='X_pca')
#     sc.tl.diffmap(adata)
#     adata.uns['iroot'] = 0  # np.where(adata.obs_names == adata.obs[idx_col].idxmin())[0][0]
#     sc.tl.dpt(adata)
#     sc.pl.pca(adata, color=['pseudotime', 'dpt_pseudotime'])
#     print(np.corrcoef(adata.obs['pseudotime'], adata.obs['dpt_pseudotime'])[0][1])
#
    # dirname = '/Users/nomo/PycharmProjects/Tree_Reconstruct_Limitations/datasets/'
    # dataset = 'prosstt_sf%d' % size_factor
    # X.to_csv(os.path.join(dirname, 'counts_%s.csv' % dataset))
#
    # meta.to_csv(os.path.join(dirname, '%s_cell_info.csv' % dataset))
#
    # D_true = pd.DataFrame(D_true, columns=X.index, index=X.index)
    # D_true.to_csv(os.path.join(dirname, 'geodesic_%s.csv' % dataset))
    print('done')
#
#     # traj = Trajectory(X, D_true)
#     # traj = Trajectory(X, D_true, meta=meta)
#     # traj.compute_tradeoff(0.002, [0.3])
