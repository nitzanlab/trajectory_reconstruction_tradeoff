import pandas as pd
import seaborn as sns


def get_color_col(meta=None, color=None, verbose=False):
    """
    Determine color column
    """
    color = 'milestone_id' if color is None else color
    if color not in meta.columns:
        raise ValueError(f'Color column {color} is not in metadata.')
    return color
    
def generate_palette(traj, color='milestone_id'):
    """
    Generate color palette for a trajectory
    """
    color = get_color_col(traj.meta, color=color)
    if color is None:
        palette = None
    else:
        unique = traj.meta[color].unique()
        palette = dict(zip(unique, sns.color_palette(n_colors=len(unique))))
        palette = palette
    return palette

def generate_palettes(trajs, **kwargs):
    """
    Given a dictionary of trajectories, returns a dictionary of color palatte for each
    """
    # generate palettes
    palettes = {}
    for traj_desc, traj in trajs.items():
        palettes[traj_desc] = generate_palette(traj, **kwargs)

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
    traj_fignames['scvelo'] = 'pancreas'
    return traj_fignames
