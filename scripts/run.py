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
    parser.add_argument('--sample', type=str, choices=['cells', 'reads', 'tradeoff', 'exp', 'tradeoffreads'], help='Sample choice')
    parser.add_argument('--repeats', type=int, default=50, help='An optional integer positional argument')

    parser.add_argument('--B', type=float, default=0.0005, help='Budget to sample')
    
    # optional, edit sampling
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    # for debugging
    # dataset = 'linear_rep0' 
    # sample = 'tradeoff' 
    # repeats = 1 
    
    dataset = args.dataset
    sample = args.sample
    repeats = args.repeats
    desc = args.desc
    datadir = args.datadir
    B = args.B
    kwargs_traj = {}
    kwargs_tradeoff = {}


    if sample == 'exp':
        # kwargs_traj['n_comp'] = 50
        #Bs = [0.0005]
        #Bs = [0.005]
        #Bs = [0.000077]
        Bs = [B]
        Pc = np.round(0.03 * 2 ** np.arange(0, 5, 0.6), 2)
        Pc = Pc[Pc < 1]
        Pt = None
        kwargs_tradeoff = {'comp_exp_corr': True, 'comp_pseudo_corr': True,'pseudo_use':'Trajectory_idx'}

    # # TEMP
    # desc = f'{desc}_original_locs'
    # kwargs_traj['do_original_locs'] = True
    # kwargs_traj['do_preprocess'] = False # only for bending map

    # TEMP
    # desc = f'{desc}_nolog1p'
    # kwargs_traj['do_log1p'] = False

    # desc = f'{desc}_dosqrt'
    # kwargs_traj['do_log1p'] = False
    # kwargs_traj['do_sqrt'] = True

    # TEMP
    # desc = f'{desc}_hvgs'
    # kwargs_traj['do_hvgs'] = True

    # load trajectory
    X, D, meta, mn = T.io.read_dataset(dataset, datadir)
    traj = T.tr.Trajectory(X, meta=meta, milestone_network=mn,**kwargs_traj)
    

    # sampling
    print(f'Sampling by {sample}')
    if sample == 'cells':
        Bs = [-1]
        #Pc = Pvar = np.round(0.01 * 2 ** np.arange(0, 6.7, 0.34), 3) # 
        Pc = Pvar = np.round(0.03 * 2 ** np.arange(0, 5, 0.3), 2)

        #Pc = Pvar = np.arange(0.05,0.95,0.05) # uniform
        Pt = Pconst = np.ones_like(Pvar)

    if sample == 'reads':
        Bs = [-1]
        Pt = Pvar = 10 ** np.arange(-6, -0.5, 0.25)
        #Pt = Pvar = 10 ** np.arange(-5, -0.5, 0.5) # TEMP
        
        #Pt = Pvar = 10 ** np.arange(-5, -0.1, 0.25); repeats = 10  #TEMP2

        #Pt = Pvar = np.arange(0.01,0.95,0.05) # uniform

        Pc = Pconst = np.ones_like(Pvar)

    if sample == 'tradeoff':
        Bs = 10 ** np.linspace(-5, -1, 10)
        Pc = Pvar = np.arange(0.01, 0.9, 0.01)
        # Pc = Pvar = np.arange(0.03, 0.6, 0.01) #TEMP
        Pt = None

    if sample == 'tradeoffreads':
        Bs = 10 ** np.linspace(-5, -1, 10)
        Pt = Pvar = np.arange(0.01, 0.9, 0.01)
        # Pc = Pvar = np.arange(0.03, 0.6, 0.01) #TEMP
        Pc = None



    L_per_traj = []
    for B in Bs:
        print(B)
        L_per_traj.append(traj.compute_tradeoff(B=B, Pc=Pc, Pt=Pt, repeats=repeats, **kwargs_tradeoff))
    L = pd.concat(L_per_traj)

    L['trajectory type'] = dataset
    
    fname = os.path.join(outdir, f'{dataset}_L_{sample}{desc}.csv')
    L.to_csv(fname)
    print(f'Successfully created {fname} of shape {L.shape}')
