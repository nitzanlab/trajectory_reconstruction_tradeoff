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
    n_comp = 10

    def __init__(self, X, D=None, meta=None, outdir=None):
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
        self.pX = Trajectory.preprocess(self.X)

        if D is None:
            D = T.ds.get_pairwise_distances(self.pX)[0]
        D = D / np.max(D)

        self.D = D
        self.meta = meta
        self.meta['original_idx'] = np.arange(self.ncells)
        self.outdir = outdir

    @staticmethod
    def set_n_comp(n_comp):
        """
        Edit number of components
        :param n_comp:
        """
        Trajectory.n_comp = n_comp

    @staticmethod
    def preprocess(X, n_comp=n_comp):
        """
        Standard preprocess
        """
        pca = PCA(n_components=n_comp, svd_solver='full')
        pX = pca.fit_transform(np.log1p(X))
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
        sX = pd.DataFrame(np.random.binomial(sX, pt), index=cellnames, columns=genenames)
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

        psX = Trajectory.preprocess(sX)
        psD, _ = T.ds.get_pairwise_distances(psX)
        return sX, psX, psD, sD, ix


    def compute_tradeoff(self, B, Pc, Pt=None, repeats=50, verbose=False, plot=False,
                         comp_pseudo_corr=False, comp_exp_corr=False, hvgs=None, n_buckets=10):
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
        Pt = [B / pc for pc in Pc] if Pt is None else Pt
        if np.any([pt < epsilon for pt in Pt]): # TODO: change condition
            print('Pt is too low')
            return

        if np.any([pc < 1/self.ncells for pc in Pc]):
            print('Restricting Pc to range of available cells')
            Pc = [pc for pc in Pc if pc > 1/self.ncells]

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
            for pc, pt in zip(Pc, Pt):
                # sample
                sX, psX, psD, sD, ix = self.subsample(pc, pt)
                smeta = self.meta.iloc[ix].copy()

                # compute error
                l1, l2, l3, lsp = T.ds.compare_distances(psD, sD)

                report = {'pc': pc, 'pt': pt, 'B': B,
                          'l1': l1, 'l2': l2, 'l3': l3, 'lsp': lsp}


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



# if __name__ == '__main__':
#     ncells = 1000
#     nsegs = 5
#     nper_seg = int(ncells / nsegs)
#     newick = '((((A:%d)B:%d)C:%d)D:%d)E:%d;' % ((nper_seg,) * nsegs)
#
#     X, D_true, meta = T.io.simulate(newick)
#     X = pd.DataFrame(X)
#
#     adata = sc.AnnData(X)
#     adata.obs = meta.loc[X.index]
#     sc.pp.log1p(adata)
#     sc.tl.pca(adata)  # recomputing since 10 pcs give disconnected pseudotime
#     sc.pp.neighbors(adata, method='gauss', use_rep='X_pca')
#     sc.tl.diffmap(adata)
#     adata.uns['iroot'] = 0  # np.where(adata.obs_names == adata.obs[idx_col].idxmin())[0][0]
#     sc.tl.dpt(adata)
#     sc.pl.pca(adata, color=['pseudotime', 'dpt_pseudotime'])
#     print(np.corrcoef(adata.obs['pseudotime'], adata.obs['dpt_pseudotime'])[0][1])
#
#     dirname = '/Users/nomo/PycharmProjects/Tree_Reconstruct_Limitations/datasets/'
#     X.to_csv(os.path.join(dirname, 'counts_prosstt.csv'))
#
#     meta.to_csv(os.path.join(dirname, 'prosstt_cell_info.csv'))
#
#     D_true = pd.DataFrame(D_true, columns=X.index, index=X.index)
#     D_true.to_csv(os.path.join(dirname, 'geodesic_prosstt.csv'))
#     print('done')
#
#     # traj = Trajectory(X, D_true)
#     # traj = Trajectory(X, D_true, meta=meta)
#     # traj.compute_tradeoff(0.002, [0.3])
