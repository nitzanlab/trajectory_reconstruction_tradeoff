import os
from pip import main
import sys; sys.path.insert(0, '../')
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
dirname = '../'
datadir = os.path.join(dirname, 'datasets')
outdir = os.path.join(dirname, 'output' )


def parse():
    """
    Parse user arguments
    """
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('dataset', type=str, help='Dataset name, e.g. hepatoblasts') 
    parser.add_argument('--datadir', type=str, default=datadir, help='Data directory')
    parser.add_argument('--outdir', type=str, default=outdir, help='Output directory')
    parser.add_argument('--desc', type=str, default='', help='Short description of run appended to filename')
    parser.add_argument('--sample', type=str, choices=['cells', 'reads', 'tradeoff', 'exp'], help='Sample choice')
    parser.add_argument('--repeats', type=int, default=50, help='An optional integer positional argument')

    parser.add_argument('--B', type=float, default=0.0005, help='Budget to sample')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    dataset = args.dataset
    sample = args.sample
    repeats = args.repeats
    desc = args.desc
    datadir = args.datadir
    B = args.B
    kwargs_traj = {}
    kwargs_tradeoff = {}


    if sample == 'exp':
        Bs = [B]
        Pc = np.round(0.03 * 2 ** np.arange(0, 5, 0.6), 2)
        Pc = Pc[Pc < 1]
        Pt = None
        kwargs_tradeoff = {'comp_exp_corr': True, 'comp_pseudo_corr': True,'pseudo_use':'Trajectory_idx'}


    # load trajectory
    X, D, meta, mn = T.io.read_dataset(dataset, datadir)
    traj = T.tr.Trajectory(X, meta=meta, milestone_network=mn,**kwargs_traj)
    

    # sampling
    print(f'Sampling by {sample}')
    if sample == 'cells':
        Bs = [-1]
        Pc = Pvar = np.round(0.03 * 2 ** np.arange(0, 5, 0.3), 2)
        Pt = Pconst = np.ones_like(Pvar)

    if sample == 'reads':
        Bs = [-1]
        Pt = Pvar = 10 ** np.arange(-6, -0.5, 0.25)
        Pc = Pconst = np.ones_like(Pvar)

    if sample == 'tradeoff':
        Bs = 10 ** np.linspace(-5, -1, 10)
        Pc = Pvar = np.arange(0.01, 0.9, 0.01)
        Pt = None

    L_per_traj = []
    for B in Bs:
        print(B)
        L_per_traj.append(traj.compute_tradeoff(B=B, Pc=Pc, Pt=Pt, repeats=repeats, **kwargs_tradeoff))
    L = pd.concat(L_per_traj)

    L['trajectory type'] = dataset
    
    fname = os.path.join(outdir, f'{dataset}_L_{sample}{desc}.csv')
    L.to_csv(fname)
    print(f'Successfully created {fname} of shape {L.shape}')