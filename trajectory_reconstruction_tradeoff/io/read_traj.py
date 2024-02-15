import os
import scanpy as sc
import pandas as pd
import numpy as np

# dictionaries shortening datasets' descriptions
shorten_dicts = {
     'hepatoblast': {'hepatoblast/hepatocyte': 'h', 'cholangiocyte': 'c',},
     'marrow': {'(Bone_Marrow_Mesenchyme)': ''},
     'thymus': {'(Thymus)': '', 'T cell_': '', 'T cell': ''},
     'muscle': {'muscle ': ''},
     'embryos': {'embryonic ': ''},
     'alpha': {'α-cell ': ''},
     'beta': {'β-cell ': ''},
     'astrocyte': {'age:': '', 'Day': 'day'},
     'rib': {'Cartilage ': '', ' high(Neonatal-Rib)':'', 'cell_': '', '_':''},
     'mesoderm': {'H7_derived_':'', 'H7_dreived':''},
     }

def read_dataset(dataset, dirname, shorten_dict=None):
    """
    Reads dataset
    :param dataset: dataset name
    :param dirname: directory name
    :param shorten_dict: dictionary for shortening dataset descriptions
    :return:
    X: expression matrix
    D: distance matrix 
    meta: metadata (dataframe with columns 'cell_id', 'milestone_id', 'group_id')
    milestone_network: milestone network (dataframe with columns 'from', 'to', 'weight')
    """
    fname_counts = os.path.join(dirname, 'counts_%s.csv' % dataset)
    fname_dists = os.path.join(dirname, 'geodesic_%s.csv' % dataset)
    fname_meta = os.path.join(dirname, '%s_cell_info.csv' % dataset)
    fname_milestone = os.path.join(dirname, '%s_milestone_network.csv' % dataset)

    fname_anndata = os.path.join(dirname, '%s.h5ad' % dataset)
    D = None
    milestone_network = None
    if os.path.isfile(fname_counts):
        X, D, meta, milestone_network = read_data_from_csv(fname_counts, fname_dists, fname_meta, fname_milestone)
    elif os.path.join(fname_anndata):
        # read from Anndata
        adata = sc.read_h5ad(os.path.join(dirname, f'{dataset}.h5ad'))
        X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        meta = adata.obs
    
    if 'group_id' in meta.columns and 'milestone_id' not in meta.columns:
            meta['milestone_id'] = meta['group_id']
    if 'branch' in meta.columns and 'milestone_id' not in meta.columns:
        meta['milestone_id'] = meta['branch']

    if shorten_dict is not None:
        milestones = meta['milestone_id'].unique()
        milestone_shorten_dict = {}
        for s in milestones:
            milestone_shorten_dict[s] = s
            for k,v in shorten_dict.items():
                if k in milestone_shorten_dict[s]:
                    milestone_shorten_dict[s] = milestone_shorten_dict[s].replace(k,v)
        meta['milestone_id'] = meta['milestone_id'].map(milestone_shorten_dict)
        milestone_network['from'] = milestone_network['from'].map(milestone_shorten_dict)
        milestone_network['to'] = milestone_network['to'].map(milestone_shorten_dict)

    return X, D, meta, milestone_network


def read_data_from_csv(fname_counts, fname_dists=None, fname_meta=None, fname_milestone=None):
    """
    Given csv file names, reads data
    :param fname_counts: expression matrix filename
    :param fname_dists: distance matrix filename
    :param fname_meta: metadata filename
    :param fname_milestone: milestone network filename
    :return:
    X: expression matrix
    D: distance matrix
    meta: metadata
    milestone_network: milestone network
    """
    X = pd.read_csv(fname_counts, index_col=0)
    X.index = X.index.astype(str)
    D = None
    meta = None
    milestone_network = None

    if os.path.isfile(fname_dists):
        D = pd.read_csv(fname_dists, index_col=0)
        D.index = D.index.astype(str)
        D = D.loc[D.columns]
        assert (all(D.columns == D.index))
        X = X.loc[D.columns]
        D = D.to_numpy()

    if os.path.isfile(fname_meta):

        meta = pd.read_csv(fname_meta, index_col=0) # TODO: was 1!! standardize
        if 'cell_id' in meta.columns:
            meta.index = meta['cell_id']
        meta.index = meta.index.astype(str)
        
        assert (meta.index == X.index).all()

    if os.path.isfile(fname_milestone):
        milestone_network = pd.read_csv(fname_milestone, index_col=0)
        # order data by milestone ordering
    else:
        print('No milestone ordering provided')
    
    return X, D, meta, milestone_network
