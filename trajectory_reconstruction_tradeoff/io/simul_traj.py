from re import L
import pandas as pd
import numpy as np
from trajectory_reconstruction_tradeoff.distances.compute_dists import get_pairwise_distances, get_pairwise_distances_branch
from prosstt import simulation as sim
from prosstt import sim_utils as sut
from prosstt import tree
from Bio import Phylo
from io import StringIO

########################################################################################################################
#TODO: better handling of this
def set_size_factor(size_factor=20):
    def calc_scalings2(cells, scale=False, scale_mean=0, scale_v=0.7):
        scalings = np.ones(cells) * size_factor
        return scalings
    sut.calc_scalings = calc_scalings2  # modify default behavior of calc_scalings function

set_size_factor()

########################################################################################################################


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

def prosstt_trajectory(newick_string, alpha=0.3, beta=2, n_resample=1, modules=30, genes=500, root='A'):
    """
    Simulate tree using Prosstt
    :return:
        expression,
        distances(in topology)
        metadata (branch, pseudotime, scalings)
    """
    #TODO: which alpha beta to use, was alpha=0, beta=3 -> alpha=0.3, beta=2 ->
    t = tree.Tree.from_newick(newick_string, modules=modules, genes=genes, density=None)

    branch_time_dict, labels = get_branch_time_dict(newick_string, root=root)
    a0 = np.min([0.05, 1 / t.modules])  # how much each gene expression program (module) contributes on average
    if len(t.branches) <= 1:
        uMs, _, _ = sim.simulate_lineage(t, a=a0)
    else:
        uMs, _, _ = sim.simulate_lineage(t, intra_branch_tol=0.5, inter_branch_tol=0, a=a0)# relative mean expression for all genes on every lineage tree branch
    base_expression = sut.simulate_base_gene_exp(t, uMs)  # simulate base gene expression
    t.add_genes(uMs, base_expression)  # scale relative expression by base expression

    X, pseudotime, branch, scalings = sim.sample_whole_tree(t, n_resample, alpha=alpha, beta=beta)

    D0 = get_pairwise_distances_branch(pseudotime, branch, branch_time_dict)
    meta = pd.DataFrame({'pseudotime': pseudotime, 'branch': branch, 'scalings': scalings})

    X = pd.DataFrame(X)
    return X, D0, meta # TODO: make D0 optional?

simulate = prosstt_trajectory # Defaults simulation to PROSSTT

# TODO: add to linear and curve metadata

def curve_trajectory(nc, R=100, frac_curve=0.5, scale_noise=5, dims=2,reverse_log1p=True):
    """
    Generate latent representation of a curve
    :param nc:
    :param R:
    :return:
    """
    nc_circ = int(nc * frac_curve)
    x_circ = None
    y_circ = None
    d0 = 1
    if nc_circ > 0:
        theta = np.linspace(0, np.pi, nc_circ)
        dtheta = theta[1] - theta[0]
        d0 = R * dtheta

        x = x_circ = R * np.sin(theta)
        y_circ = R * np.cos(theta)
        y = y_circ = y_circ + np.abs(y_circ.min())

    # x = np.hstack((np.linspace(-314, 0, nc), x))
    # y = np.hstack((np.linspace(100, 100, nc), y))
    nc_line = nc - nc_circ
    x_line = None
    y_line = None
    
    if nc_line > 0:
        x = x_line = np.linspace(-d0 * nc_line, 0, nc_line)
        y = y_line = np.linspace(R, R, nc_line)     
        
    if (x_line is not None) and (x_circ is not None):
        x = np.hstack((x_line, x_circ))
        y = np.hstack((y_line, y_circ))

    x = x + np.random.poisson(scale_noise, size=nc)
    y = y + np.random.poisson(scale_noise, size=nc)
    pX = np.vstack((x, y)).T
    if dims > 2:
        pX_mapping = np.random.poisson(scale_noise, size=(dims, 2))
        pX_mapping = pX_mapping / np.linalg.norm(pX_mapping)
        # pX_noise = np.random.normal(0, scale=scale_noise, size=(nc, dims-2))
        # pX = np.hstack((pX, pX_noise))
        pX = (pX_mapping @ pX.T).T
        # pX_noise = np.random.poisson(scale_noise, size=(nc, dims-2))
        # pX = np.hstack((pX, pX_noise))
    if reverse_log1p:
        pX = np.exp(pX) - 1
        pX = np.maximum(pX, 0) #TODO: remove?
    
    return pX

# def curve_clusters_trajectory(nc, k=5, p_in_k=0.7, R=100, scale_noise=5, scale_cluster=10, dims=2, reverse_log1p=False):
#     """
#     Generate latent representation of a curve
#     :param nc:
#     :param k: number of clusters
#     :param p_in_k: fraction of cells in clusters
#     :param R:
#     :return:
#     """
#     nc_in_k = int(nc/k * p_in_k)
#     nc_out_k = nc - nc_in_k*k
#     nc_circ = nc_out_k + k
#     theta = np.linspace(0, np.pi, nc_circ)
#     dtheta = theta[1] - theta[0]
#     d0 = R * dtheta
#     x_circ = R * np.sin(theta)
#     y_circ = R * np.cos(theta)
    
#     x = x_circ + np.random.normal(0, scale=scale_noise, size=nc_circ)
#     y = y_circ + np.random.normal(0, scale=scale_noise, size=nc_circ)
    
#     x_cents = x_circ[::int(nc_circ / (k-1))]
#     y_cents = y_circ[::int(nc_circ / (k-1))]

#     for i in np.arange(k):
#         x_cluster = np.random.normal(x_cents[i], scale=scale_cluster, size=nc_in_k-1)
#         y_cluster = np.random.normal(y_cents[i], scale=scale_cluster, size=nc_in_k-1)
        
#         x = np.hstack((x, x_cluster))
#         y = np.hstack((y, y_cluster))

#     pX = np.vstack((x, y)).T
#     if dims > 2:
#         pX_mapping = np.random.poisson(scale_noise, size=(2, dims))
#         # pX_noise = np.random.normal(0, scale=scale_noise, size=(nc, dims-2))
#         # pX = np.hstack((pX, pX_noise))
#         pX = pX_mapping @ pX
#     if reverse_log1p:
#         pX = np.exp(pX) - 1
#         pX = np.maximum(pX, 0) #TODO: remove?
#     return pX


def line_trajectory(nc, endpoint1=(10,0), endpoint2=(0,10), std=1, nonneg=True):
    """
    """
    t = (np.arange(nc) / nc).reshape(-1, 1)

    endpoint1 = np.array(endpoint1).reshape(1, -1)
    endpoint2 = np.array(endpoint2).reshape(1, -1)
    ngenes = np.maximum(len(endpoint1), len(endpoint2))

    X = (1-t) * endpoint1 + (t) * endpoint2 + np.random.normal(scale=std, size=(nc, ngenes))
    if nonneg:
        X = np.maximum(X,0)
    return X


if __name__ == '__main__':
    newick_string = '(((A:250)B:250)C:250)D:250;'
    X, D0, meta = prosstt_trajectory(newick_string=newick_string, return_meta=True)
