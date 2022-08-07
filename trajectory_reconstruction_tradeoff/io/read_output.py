import os
import pandas as pd

def read_output(datasets, outdir, sample, exp_desc=''):
    """
    Reads saved compute_tradeoff output. 
    sample - describes what was subsampled. Can be: reads/cells/tradeoff

    """
    L_dict = {}

    for dataset in datasets:
        fname = os.path.join(outdir, f'{dataset}_L_{sample}{exp_desc}.csv')
        if os.path.isfile(fname):
            L = pd.read_csv(fname, index_col=0)# TEMP
            L_dict[dataset] = L
    
    return L_dict