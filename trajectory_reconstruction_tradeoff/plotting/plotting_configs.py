import pandas as pd
import seaborn as sns



# Includes specific configurations of datasets tested here

color_col_names = ['branch', 'milestone_id']


colors_simul = {
    'curve': 'orange',
    'linear_rep0': 'navy', 
    'bifur_at_1': 'blue', 
    'bifur_at_2': 'deepskyblue',#'cyan', 
    'bifur_at_3': 'aquamarine', 
    'bifur_at_1_to_3': 'lightsalmon', 
    'bifur_at_2_to_3': 'tomato', 
    'bifur_at_1_to_4': 'red', 
    }

colors_real = {  
    'hayashi': 'seagreen',
    'dendritic': 'mediumseagreen',
    'hepatoblast': 'plum',
    'fibroblasts': 'orchid', 
    'hematopoiesis': 'mediumorchid',
    }


def is_prosstt(dataset):
    """
    From saved datasets, returns true if was simulated with PROSSTT
    """
    return dataset in colors_simul.keys()

colors_datasets = {**colors_simul, **colors_real}



def get_color_col(meta=None, color_col=None, verbose=False):
    """
    Determine color column
    """
    if not isinstance(meta, pd.DataFrame):
        if verbose:
            print(f'metadata is of type {type(meta)}. metadata has to be a dataframe')
        return None
    color = color_col if color_col else list(set(meta.columns).intersection(color_col_names))
    if len(color) != 1:
        if verbose:
            print(f'{len(color)} color columns were found.')
        return None
    color = color[0]
    if color not in meta.columns:
        if verbose:
            print(f'Color column {color} is not in metadata.')
        return None
    return color
    

def generate_palettes(trajs, color_col=None):
    """
    Given a dictionary of trajectories, returns a dictionary of color palatte for each
    """
    # generate palettes
    palettes = {}
    for traj_desc, traj in trajs.items():
        color = get_color_col(traj.meta, color_col=color_col)
        if color is None:
            palettes[traj_desc] = None
        else:
            unique = traj.meta[color].unique()
            palette = dict(zip(unique, sns.color_palette(n_colors=len(unique))))
            palettes[traj_desc] = palette

    return palettes

def get_plot_configs(datasets, in_one_row=True):
    """
    Sets location for each dataset in plot
    """
    ntrajs = len(datasets)
    one_row_config = {'nrows': 1, 'ncols': ntrajs, 'figsize': (6*ntrajs, 6), 'constrained_layout': True, 'tight_layout': False, 'dpi': 300}
    one_loc = {traj_desc: itraj for itraj,traj_desc in enumerate(datasets) }

    # plotting configs - one row location
    plot_config = one_row_config
    plot_loc = one_loc
    return plot_config, plot_loc

def get_traj_fignames(datasets):
    """
    Renaming of datasets for plots
    """
    traj_fignames = {k:k for k in datasets}
    traj_fignames['hayashi'] = 'mESC'
    traj_fignames['linear_rep0'] = 'simulated'
    return traj_fignames

# color_map = {'A': '#e6194b', 'B': '#3cb44b', 'C': '#ffe119', 'D': '#4363d8', 'E': '#f58231', 'F': '#911eb4',
#              'G': '#46f0f0', 'H': '#f032e6', 'I': '#bcf60c', 'J': '#fabebe', 'K': '#008080', 'L': '#e6beff',
#              'M': '#9a6324', 'N': '#fffac8', 'O': '#800000', 'P': '#aaffc3', 'Q': '#808000', 'R': '#ffd8b1',
#              'S': '#000075', 'T': '#808080'}
