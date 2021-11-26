import numpy as np
import pandas as pd
import plotting as P
from sklearn.decomposition import PCA
from compute_dists import get_pairwise_distances, compare_distances

epsilon = 10e-5

def subsample(X, D_true, pc, pt, n_pc=10):
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
    X = X.to_numpy() if isinstance(X, pd.core.frame.DataFrame) else X
    N = X.shape[0]
    n = int(N * pc)
    ix = np.random.choice(N, n, replace=False)
    X1 = X[ix, :]
    X1 = X1.astype(int)
    X1 = np.random.binomial(X1, pt)
    D0 = D_true[ix][:, ix]  # subsampled ground truth pairwise distances
    D0 = D0 / np.max(D0)

    # compute distances of subsample
    pca = PCA(n_components=n_pc, svd_solver='full')
    expr_red1 = pca.fit_transform(np.log1p(X1))
    D1, _ = get_pairwise_distances(expr_red1)
    return X1, expr_red1, D1, D0, ix

def compute_tradeoff(X, D_true, B, Pc, Pt=None, n_pc=10, repeats=50, verbose=False, plot=False):
    """
    Compute reconstruction error for subsampled data within budget tradeoff
    :param X: counts data
    :param D_true: geodesic distances
    :param B: counts budget
    :param Pc: cell capture probabilities
    :param n_pc: number of PCs
    :param repeats: number of repeats
    :param verbose: print messages
    :param plot: plot reduced expression for each set of downsample params
    :return:
        dataframe with sampling params and errors
    """
    Pt = [B / pc for pc in Pc] if Pt is None else Pt
    if np.any([pt < epsilon for pt in Pt]):
        print('Pt is too low')
        return

    L = []
    X = X.to_numpy() if isinstance(X, pd.core.frame.DataFrame) else X
    X = X.astype(int)

    for k in range(repeats):
        if verbose:
            print(k)
        for pc, pt in zip(Pc, Pt):
            # sample
            X1, expr_red1, D1, D0, _ = subsample(X, D_true, pc, pt, n_pc=n_pc)

            # compute error
            l1, l2, l3, lsp = compare_distances(D1, D0)
            L.append({'pc': pc, 'pt': pt, 'B': B,
                      'l1': l1, 'l2': l2, 'l3': l3, 'lsp': lsp})

            if plot and (k == 0):
                tit = 'B = %.5f, pc = %.5f, pt = %.5f \n l1 = %.2f, l2 = %.2f' % (B, pc, pt, l1, l2)
                P.plot_pca2d(expr_red1, title=tit)

    L = pd.DataFrame(L)
    return L


def infer_complex_pt(pc1, pc2, B1, B2):
    """
    Infer pt from pairs of (budget, optimal cell capture)
    :return:
    """
    alpha = pc2 / pc1
    K = B2 / B1
    pt = np.exp((1 - alpha) / np.log(K / alpha))
    return pt


def infer_optimal_pc(B, B1, pc1, B2, pc2, traj_type='simple'):
    """
    Infer the optimal pc for a new experiment
    :param B: new budget
    :param B1:
    :param pc1:
    :param B2:
    :param pc2:
    :param traj_type: 'simple' or 'complex' depending on curvatures and bifurcations
    :return:
    """
    if traj_type == 'simple':
        return np.mean((pc1, pc2))
    if traj_type == 'complex':
        # TODO: probably incorrect, need to change
        pt = infer_complex_pt(pc1, pc2, B1, B2)
        return B / pt
