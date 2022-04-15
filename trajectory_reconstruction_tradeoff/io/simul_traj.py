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

def simulate(newick_string, alpha=0.3, beta=2, n_resample=1, modules=30, genes=500, root='A'):
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

    D0, _ = get_pairwise_distances_branch(pseudotime, branch, branch_time_dict)
    meta = pd.DataFrame({'pseudotime': pseudotime, 'branch': branch, 'scalings': scalings})

    X = pd.DataFrame(X)
    return X, D0, meta



def curve_trajectory(nc, R=100):
    """
    Generate latent representation of a curve
    :param nc:
    :param R:
    :return:
    """
    nc = int(nc/2)
    theta = np.linspace(0, np.pi, nc)
    dtheta = theta[1] - theta[0]
    d0 = R * dtheta

    x = R * np.sin(theta)
    y = R * np.cos(theta)

    # x = np.hstack((np.linspace(-314, 0, nc), x))
    # y = np.hstack((np.linspace(100, 100, nc), y))
    x = np.hstack((np.linspace(-nc, 0, nc), x))  # setting constant dist 1 for linear part
    y = np.hstack((np.linspace(R, R, nc), y))
    pX = np.vstack((x, y)).T
    return pX

if __name__ == '__main__':
    newick_string = '(((A:250)B:250)C:250)D:250;'
    X, D0, meta = simulate(newick_string=newick_string, return_meta=True)
