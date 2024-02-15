import os
import pandas as pd

def read_output(dataset, outdir, sample, exp_desc=''):
    """
    Reads saved results output.
    :param dataset: dataset name
    :param outdir: output directory
    :param sample: type of subsampling experiment
    :param exp_desc: description of the experiment
    :return: 
    L: saved subsampling results
    """
    fname = os.path.join(outdir, f'{dataset}_L_{sample}{exp_desc}.csv')
    if os.path.isfile(fname):
        L = pd.read_csv(fname, index_col=0)

    return L

def read_outputs(datasets, **kwargs):
    """
    Reads saved compute_tradeoff output. 
    :param datasets: list of datasets
    :return: dictionary of dataframes
    """
    L_dict = {}

    for dataset in datasets:
        L_dict[dataset] = read_output(dataset, **kwargs)
    
    return L_dict