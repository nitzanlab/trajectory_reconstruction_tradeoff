import os
import pandas as pd
import numpy as np

def read_dataset(dataset, dirname):
    """

    :param dataset:
    :param dirname:
    :return:
    """
    fname_counts = os.path.join(dirname, 'counts_%s.csv' % dataset)
    fname_dists = os.path.join(dirname, 'geodesic_%s.csv' % dataset)
    fname_meta = os.path.join(dirname, '%s_cell_info.csv' % dataset)
    fname_milestone = os.path.join(dirname, '%s_milestone_network.csv' % dataset)
    return read_data(fname_counts, fname_dists, fname_meta, fname_milestone)

def read_data(fname_counts, fname_dists=None, fname_meta=None, fname_milestone=None):
    """
    Read and prepare counts and distance matrices
    :param fname_counts:
    :param fname_dists:
    :return:
        expression,
        distances(read from file), if available
    """
    X = pd.read_csv(fname_counts, index_col=0)
    X.index = X.index.astype(str)
    D = None
    meta = None
    if fname_dists:
        D = pd.read_csv(fname_dists, index_col=0)
        D.index = D.index.astype(str)
        D = D.loc[D.columns]
        assert (all(D.columns == D.index))
        X = X.loc[D.columns]
        D = D.to_numpy()
    if fname_meta:
        meta = pd.read_csv(fname_meta, index_col=1) # TODO: was 1!! standardize
        meta.index = meta.index.astype(str)
        assert (meta.index == X.index).all()
    # TODO: convert milestone network to metadata?
    return X, D, meta

if __name__ == '__main__':
    dirname = '/Users/nomo/PycharmProjects/Tree_Reconstruct_Limitations/datasets/'  # TODO: change to relative path
    fname = 'hepatoblast'
    fname_counts = os.path.join(dirname, 'counts_' + fname + '.csv')
    fname_dists = os.path.join(dirname, 'geodesic_' + fname + '.csv')
    X, D = read_data(fname_counts=fname_counts, fname_dists=fname_dists)
