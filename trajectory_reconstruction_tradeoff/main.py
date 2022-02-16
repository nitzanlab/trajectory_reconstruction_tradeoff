from trajectory_reconstruction_tradeoff import io as io
from trajectory_reconstruction_tradeoff import opt as opt
from trajectory_reconstruction_tradeoff.trajectory import Trajectory
import trajectory_reconstruction_tradeoff.plotting.plotting as P
import matplotlib.pyplot as plt
import os



if __name__ == '__main__':

    # # read data
    # dirname = '/Users/nomo/PycharmProjects/Tree_Reconstruct_Limitations/datasets/' # TODO: change to relative path
    # fname = 'hepatoblast'
    # fname_counts = os.path.join(dirname, 'counts_' + fname + '.csv')
    # fname_dists = os.path.join(dirname, 'geodesic_' + fname + '.csv')
    # X, D = io.read_data(fname_counts=fname_counts, fname_dists=fname_dists)

    n_per_branch = 100
    newick_string = "(C:%d,B:%d)A:%d;" % (n_per_branch, n_per_branch, n_per_branch)
    X, D, meta = io.simulate(newick_string)

    traj = Trajectory(X, D)

    P.plot_pca2d(traj.pX, meta, color='branch')
    P.plot_pca2d(traj.pX, meta, color='pseudotime')

    Pc = [0.05, 0.1, 0.2, 0.23, 0.24, 0.25, 0.7, 0.8, 0.97]
    # Pc = [0.1, 0.25, 0.8]
    repeats = 3

    # compute opt curve
    B1 = 0.00008
    L1 = traj.compute_tradeoff(B1, Pc, repeats=repeats, plot=False)
    pc_opt1 = opt.find_min_nc(L1)
    fig, ax = plt.subplots(figsize=(14, 10))
    P.plot_tradeoff(L1, color_mean='red', color_std='red', pc_opt=pc_opt1, ax=ax)

    B2 = 0.0004
    L2 = traj.compute_tradeoff(B2, Pc, repeats=repeats, plot=False)
    pc_opt2 = opt.find_min_nc(L2)
    P.plot_tradeoff(L2, pc_opt=pc_opt2, ax=ax)
    plt.show()

    # given a new budget, infer optimal cell capture probability
    B = 0.0002
    pc_simple = opt.infer_optimal_nc(B=B, B1=B1, B2=B2, nc1=pc_opt1, nc2=pc_opt2, traj_type='simple')
    pc_complex = opt.infer_optimal_nc(B=B, B1=B1, B2=B2, nc1=pc_opt1, nc2=pc_opt2, traj_type='complex')
    print('For B=%.5f: \n pc=%.05f for simple trajectory \n pc=%.05f for complex trajectory' % (B, pc_simple, pc_complex))
