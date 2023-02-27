
import sys; sys.path.insert(0, '..')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import trajectory_reconstruction_tradeoff as T
import random
import scanpy as sc

random.seed(0)

# create a pilot dataset
dirname = '/Users/nomo/PycharmProjects/trajectory_reconstruction_tradeoff/'
datadir = os.path.join(dirname, 'datasets')
outdir = os.path.join(dirname, 'output')

desc = 'uppilot'


save = True

################################################ hayashi #######################################################
# create subsample
dataset = 'hayashi'
pc = 0.1
pt = 0.1 
X, _, meta, mn = T.io.read_dataset(dataset=dataset, dirname=datadir)
traj = T.tr.Trajectory(X, meta=meta)

# subsample 10% of cells
sX_subcells, _, _, _, _, _, ix_subcells, _ = traj.subsample(pc, 1)
smeta_subcells = meta.iloc[ix_subcells]

# subsample 10% of reads
sX_subreads, _, _, _, _, _, ix_subreads, _ = traj.subsample(1, pt)
smeta_subreads = meta.iloc[ix_subreads]

# combine their sampling to get the pilot experiment
cells = list(sX_subcells.index)
remaining_cells = list(X.index.difference(cells))
sX_subreadcells = sX_subreads.loc[cells]
smeta_subreadcells = meta.loc[cells]

if save:
    sX_subreads.to_csv(os.path.join(outdir, f'counts_{dataset}_{desc}subreads.csv'))
    smeta_subreads.to_csv(os.path.join(outdir, f'{dataset}_{desc}subreads_cell_info.csv'))

    sX_subcells.to_csv(os.path.join(outdir, f'counts_{dataset}_{desc}subcells.csv'))
    smeta_subcells.to_csv(os.path.join(outdir, f'{dataset}_{desc}subcellss_cell_info.csv'))

    sX_subreadcells.to_csv(os.path.join(outdir, f'counts_{dataset}_{desc}.csv'))
    smeta_subreadcells.to_csv(os.path.join(outdir, f'{dataset}_{desc}_cell_info.csv'))


################################################ hayashi #######################################################

# trajectory of pilot
straj = T.tr.Trajectory(sX_subreadcells, meta=smeta_subreadcells)
D = straj.D # considering pilot distance matrix as GT
dmax_D = np.max(D); nD = D / dmax_D
straj_subcells = T.tr.Trajectory(sX_subcells, meta=smeta_subcells); straj_subcells.meta['idx'] = np.arange(len(straj_subcells.meta))
straj_subreads = T.tr.Trajectory(sX_subreads, meta=smeta_subreads); straj_subreads.meta['idx'] = np.arange(len(straj_subreads.meta))



# subsample cells from sX_subreads
B = -1

Pc = Pvar = np.arange(0.1, 0.91, 0.1) #np.round(0.03 * 2 ** np.arange(0, 5, 0.3), 2)
Pt = Pconst = np.ones_like(Pvar)
repeats = 2
L = []
for (pc, pt) in zip(Pc, Pt):
    for _ in range(repeats):
        sampled_cells = cells + list(np.random.choice(remaining_cells, size=int(pc * len(remaining_cells)), replace=False))
        
        ix_sampled_cells = straj_subreads.meta.loc[sampled_cells]['idx']
        sX, psX, lsX, psD, sD, psP, ix, pca = straj_subreads.subsample(pc, pt, ix=ix_sampled_cells)
        
        # subset cells from psD
        ix_cells = np.arange(len(cells))
        psD = psD[ix_cells][:, ix_cells]

        # report = straj.evaluate(*subsample_result, pc=pc, pt=pt)
        nc = sX.shape[0]
        nr = sX.sum(1).mean()
        
        # smeta = self.meta.iloc[ix].copy()

        dmax_psD = np.max(psD); npsD = psD / dmax_psD
        # dmax_sD = np.max(sD); nsD = sD / dmax_sD

        # compute error
        l1, l2, ldist, fcontrac, fexpand, lsp = T.ds.compare_distances(nD, npsD)

        
        report = {'nc': nc, 'nr': nr, 'Br': sX.sum().sum(),
                  'l1': l1, 'l2': l2, 'ldist': ldist, 'fcontrac': fcontrac, 'fexpand': fexpand, 'lsp': lsp, 
                  'dmax_psD': dmax_psD, 'dmax_sD': dmax_D}


                            
        report = {'pc': 1 + pc, 'pt': pt, 'B': B, 
                    'log pc': np.log(1+pc), 'log pt': np.log(pt), 
                    'sqrt inv pc': np.sqrt(1/(1+pc)), 'sqrt inv pt': np.sqrt(1/pt), 
                    **report}
        L.append(report)

L = pd.DataFrame(L)

if save:
    L.to_csv(os.path.join(outdir, f'{dataset}_L_cells{desc}.csv'))



# subsample reads from sX_subcells
# subsample cells from sX_subreads
B = -1

Pt = Pvar = np.arange(0.1, 0.91, 0.1) #np.round(0.03 * 2 ** np.arange(0, 5, 0.3), 2)
Pc = Pconst = np.ones_like(Pvar)
repeats = 2
L = []
for (pc, pt) in zip(Pc, Pt):
    for _ in range(repeats):
        sX, psX, lsX, psD, sD, psP, ix, pca = straj_subcells.subsample(pc, pt)
        
        
        # report = straj.evaluate(*subsample_result, pc=pc, pt=pt)
        nc = sX.shape[0]
        nr = sX.sum(1).mean()
        
        # smeta = self.meta.iloc[ix].copy()

        dmax_psD = np.max(psD); npsD = psD / dmax_psD
        # dmax_sD = np.max(sD); nsD = sD / dmax_sD

        # compute error
        l1, l2, ldist, fcontrac, fexpand, lsp = T.ds.compare_distances(nD, npsD)

        
        report = {'nc': nc, 'nr': nr, 'Br': sX.sum().sum(),
                  'l1': l1, 'l2': l2, 'ldist': ldist, 'fcontrac': fcontrac, 'fexpand': fexpand, 'lsp': lsp, 
                  'dmax_psD': dmax_psD, 'dmax_sD': dmax_D}


                            
        report = {'pc': pc, 'pt': 1 + pt, 'B': B, 
                    'log pc': np.log(pc), 'log pt': np.log(1 + pt), 
                    'sqrt inv pc': np.sqrt(1/(pc)), 'sqrt inv pt': np.sqrt(1/(1 + pt)), 
                    **report}
        L.append(report)

L = pd.DataFrame(L)

if save:
    L.to_csv(os.path.join(outdir, f'{dataset}_L_reads{desc}.csv'))


# copy milestone network file

# mn.to_csv(os.path.join(outdir, f'{dataset}_{desc}_milestone_network.csv'), index=False)