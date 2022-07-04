import os
from pip import main
import sys; sys.path.insert(0, '/cs/usr/nomoriel/PycharmProjects/trajectory_reconstruction_tradeoff')
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import trajectory_reconstruction_tradeoff as T
import random
import scanpy as sc
import altair as alt
from altair_saver import save

random.seed(20)

# trajectory params
dirname = '/cs/labs/mornitzan/nomoriel/trajectory_reconstruction_tradeoff/'
datadir = os.path.join(dirname, 'datasets')
outdir = os.path.join(dirname, 'output_new2' )


def parse():
    """
    Parse user arguments
    """
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('dataset', type=str, help='Dataset name, e.g. hepatoblasts')
    parser.add_argument('--datadir', type=str, default=datadir, help='Data directory')
    parser.add_argument('--outdir', type=str, default=outdir, help='Output directory')
    
    parser.add_argument('--sample', type=str, choices=['cells', 'reads', 'tradeoff'], help='Sample choice')

    parser.add_argument('--repeats', type=int, default=50, help='An optional integer positional argument')
    
    # optional, edit sampling
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    # load trajectory
    dataset = args.dataset
    X, D, meta = T.io.read_dataset(dataset, args.datadir)
    traj = T.tr.Trajectory(X, meta=meta)
    

    # sampling
    sample = args.sample

    if sample == 'cells':
        Bs = [-1]
        Pc = Pvar = np.round(0.01 * 2 ** np.arange(0, 6.7, 0.34), 3) # np.round(0.03 * 2 ** np.arange(0, 5, 0.3), 2)
        Pt = Pconst = np.ones_like(Pvar)

    if sample == 'reads':
        Bs = [-1]
        Pt = Pvar = 10 ** np.arange(-6, -0.5, 0.25)
        Pc = Pconst = np.ones_like(Pvar)

    if sample == 'tradeoff':
        Bs = 10 ** np.linspace(-5, -1, 10)
        Pc = Pvar = np.arange(0.01, 0.9, 0.01)
        Pt = -1


    L_per_traj = []
    for B in Bs:
        print(B)
        L_per_traj.append(traj.compute_tradeoff(B=B, Pc=Pc, Pt=Pt, repeats=args.repeats, comp_deltas=True))
    L = pd.concat(L_per_traj)

    L['trajectory type'] = dataset

    L.to_csv(os.path.join(outdir, f'{dataset}_L_{sample}.csv'))
    print(L.shape)
