import os
import pandas as pd

def read_output(dataset, outdir, sample, exp_desc=''):
    fname = os.path.join(outdir, f'{dataset}_L_{sample}{exp_desc}.csv')
    if os.path.isfile(fname):
        L = pd.read_csv(fname, index_col=0)# TEMP

    return L

def read_outputs(datasets, **kwargs):
    """
    Reads saved compute_tradeoff output. 
    sample - describes what was subsampled. Can be: reads/cells/tradeoff

    """
    L_dict = {}

    for dataset in datasets:
        L_dict[dataset] = read_output(dataset, **kwargs)
    
    return L_dict