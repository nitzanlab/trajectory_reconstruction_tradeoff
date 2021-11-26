import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from compute_dists import get_pairwise_distances, get_pairwise_distances_branch
from prosstt import simulation as sim
from prosstt import sim_utils as sut
from prosstt import tree
from Bio import Phylo
from io import StringIO

def preprocess(X, n_pc=10):
    """
    Standard preprocess
    """
    pca = PCA(n_components=n_pc, svd_solver='full')
    expr_red = pca.fit_transform(np.log1p(X))
    return expr_red


def get_branch_time_dict(newick_string, root='A'):
    """
    Generate dictionary of branches and number of timepoints in between
    """
    tree = Phylo.read(StringIO(newick_string), "newick")
    labels = [x.name for x in tree.find_clades()]
    labels_choose2 = [(l1, l2) for l1 in labels for l2 in labels]
    branch_time_dict = {}
    for l1, l2 in labels_choose2:
        if root not in [l1, l2] and not set(tree.get_path(l1)).issubset(set(tree.get_path(l2))) and not set(tree.get_path(l2)).issubset(set(tree.get_path(l1))):
            branch_time_dict[(l1, l2)] = tree.trace(root, root)[0].branch_length + tree.distance(tree.common_ancestor(l1, l2).name)
            branch_time_dict[(l2, l1)] = tree.trace(root, root)[0].branch_length + tree.distance(tree.common_ancestor(l2, l1).name)
    return branch_time_dict, labels

def simulate(newick_string, alpha=0.3, beta=2, n_pc=10, n_resample=1, modules=30, genes=500, root='A'):
    """
    Simulate tree using Prosstt
    :return:
        expression,
        reduced expression,
        distances(in expression),
        distances(in topology)
    """

    t = tree.Tree.from_newick(newick_string, modules=modules, genes=genes, density=None)

    branch_time_dict, labels = get_branch_time_dict(newick_string, root=root)
    a0 = np.min([0.05, 1 / t.modules])  # how much each gene expression program (module) contributes on average
    uMs, _, _ = sim.simulate_lineage(t, a=a0, intra_branch_tol=-1,
                                     inter_branch_tol=0)  # relative mean expression for all genes on every lineage tree branch
    base_expression = sut.simulate_base_gene_exp(t, uMs)  # simulate base gene expression
    t.add_genes(uMs, base_expression)  # scale relative expression by base expression

    X, pseudotime, branch, scalings = sim.sample_whole_tree(t, n_resample, alpha=alpha, beta=beta)

    D0, _ = get_pairwise_distances_branch(pseudotime, branch, branch_time_dict)
    expr_red = preprocess(X, n_pc=n_pc)
    D, _ = get_pairwise_distances(expr_red)

    return X, expr_red, D, D0


def read_data(fname_counts, fname_dists=None, n_pc=10):
    """
    Read and prepare counts and distance matrices
    :param fname_counts:
    :param fname_dists:
    :param n_pc: number of PCs
    :return:
        expression,
        reduced expression,
        distances(in expression),
        distances(read from file), if available
    """
    X = pd.read_csv(fname_counts, index_col=0)
    expr_red = preprocess(X, n_pc=n_pc)
    D, _ = get_pairwise_distances(expr_red)
    D0 = None
    if fname_dists:
        df = pd.read_csv(fname_dists, index_col=0)
        df = df.loc[df.columns]
        assert (all(df.columns == df.index))
        X = X.loc[df.columns]
        D0 = df.to_numpy()
        D0 = D0 / np.max(D0)

    return X, expr_red, D, D0

# def read_data_edvin(fname, n_pc=10, dirname=''):
#     """
#     Read data, assuming format as in Saelen et al.
#     :param fname:
#     :param color:
#     :param n_pc: number of PCs
#     :param dirname:
#     :return:
#     """
#     df = pd.read_csv(os.path.join(dirname, 'geodesic_' + fname + '.csv'), index_col=0)
#     cellnames = df.columns
#     df = df.loc[cellnames]
#     assert (all(cellnames == df.index))
#     D0 = df.to_numpy()
#     D0 = D0 / np.max(D0)
#
#     # Edvin? current:
#     expr = pd.read_csv(os.path.join(dirname, 'expr_' + fname + '.csv'), index_col=0)
#     expr = expr.loc[cellnames]
#     pca = PCA(n_components=n_pc, svd_solver='full')
#     expr_red = pca.fit_transform(expr)
#
#     # alternative
#     X = pd.read_csv(os.path.join(dirname, 'counts_' + fname + '.csv'), index_col=0)
#     X = X.loc[cellnames]
#     # expr_red = preprocess(X)
#     #    # X = 2**expr - 1
#     D, _ = get_pairwise_distances(expr_red)
#
#     return X, expr_red, D, D0
