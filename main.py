import numpy as np
from trajectories import read_data, simulate
from tradeoff_curve import compute_tradeoff, infer_optimal_pc
import plotting as P
import matplotlib.pyplot as plt
import os



if __name__ == '__main__':

    # read data
    dirname = '/Users/nomo/PycharmProjects/Tree_Reconstruct_Limitations/datasets/' # TODO: change to relative path
    fname = 'hepatoblast'
    fname_counts = os.path.join(dirname, 'counts_' + fname + '.csv')
    fname_dists = os.path.join(dirname, 'geodesic_' + fname + '.csv')
    X, expr_red, D, D0 = read_data(fname_counts=fname_counts, fname_dists=fname_dists)

    # n_per_branch = 100
    # newick_string = "(C:%d,B:%d)A:%d;" % (n_per_branch, n_per_branch, n_per_branch)
    # X, expr_red, D, D0 = simulate(newick_string)

    P.plot_pca2d(expr_red)

    Pc = [0.05, 0.1, 0.2, 0.23, 0.24, 0.25, 0.7, 0.8, 0.97]
    # Pc = [0.1, 0.25, 0.8]
    repeats = 10

    # compute tradeoff curve
    B1 = 0.0001
    L1 = compute_tradeoff(X, D0, B1, Pc, repeats=repeats, plot=True)
    fig, ax = plt.subplots(figsize=(14, 10))
    pc_opt1 = P.plot_tradeoff(L1, color_mean='red', color_std='red', plot_min=True, ax=ax)

    B2 = 0.0003
    L2 = compute_tradeoff(X, D0, B2, Pc, repeats=repeats, plot=True)
    pc_opt2 = P.plot_tradeoff(L2, plot_min=True, ax=ax)
    plt.show()

    # given a new budget, infer optimal cell capture probability
    B = 0.0002
    pc_simple = infer_optimal_pc(B=B, B1=B1, B2=B2, pc1=pc_opt1, pc2=pc_opt2, traj_type='simple')
    pc_complex = infer_optimal_pc(B=B, B1=B1, B2=B2, pc1=pc_opt1, pc2=pc_opt2, traj_type='complex')
    print('For B=%.5f: \n pc=%.05f for simple trajectory \n pc=%.05f for complex trajectory' % (B, pc_simple, pc_complex))
