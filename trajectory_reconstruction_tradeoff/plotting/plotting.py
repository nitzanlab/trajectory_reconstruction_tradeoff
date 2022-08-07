
import numpy as np
import pandas as pd
import altair as alt
import matplotlib
import gif
import ipywidgets as widgets
from ipywidgets import interact
from collections import OrderedDict
import matplotlib.pyplot as plt
from trajectory_reconstruction_tradeoff.opt import find_min_nc
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from matplotlib.collections import EllipseCollection
import seaborn as sns
from .plotting_configs import get_color_col

plt.rcParams.update({'figure.max_open_warning': 0})
titlesize = 35
labelsize = 30
ticksize = 25
legendsize = 28


def plot_pca2d(pX, meta=None, color=None, title='', fname=None, ax=None,
               xlabel = 'PC1', ylabel = 'PC2', colorlabel=None, legend=True, legendsize=legendsize, titlesize=titlesize, **kwargs):
    """
    Plot expression in reduced space
    :param pX: expression reduced representation (cells x reduced dim)
    :param meta: meta data (cells x meta dim)
    :param color: meta data column to color by
    :param title: title of plot
    :param fname: file name to save plot to
    :param ax: axis to plot on
    :param colorlabel: label for color
    :param legend: whether to show legend
    :param kwargs: additional arguments to pass to sns.scatterplot
    """
    
    ax_none = ax is None
    if ax_none:
        fig, ax = plt.subplots(figsize=(6,6))
    
    pX = pX.values if isinstance(pX, pd.DataFrame) else pX
    
    df = pd.DataFrame({xlabel: pX[:,0], ylabel: pX[:,1]})

    color = get_color_col(meta, color_col=color)
    colorlabel = color.title() if (colorlabel is None) and color else colorlabel
    

    if (meta is not None) and (color):# in meta.columns):
        df[colorlabel] = meta[color].values
        kwargs['hue'] = colorlabel if 'hue' not in kwargs.keys() else kwargs['hue']

    sns.scatterplot(data=df, x=xlabel, y=ylabel, ax=ax, **kwargs)
    ax.set_title(title, fontsize=titlesize)
    ax.axis('off')

    # Add a legend
    if color and legend:
        ax.legend(fontsize=legendsize)
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
        ax.legend(handletextpad=0.01,
                  loc='lower center', 
                  bbox_to_anchor=(0.5, -0.5),
                  ncol=2, fontsize=20, frameon=False)
    elif color and not legend:
        ax.get_legend().remove()

    
    if fname is not None:
        plt.savefig(fname)
    elif ax_none:
        plt.show()



def plot_tradeoff(L, xcol='pc', ycol='l1', xlabel='Sampling probability', ylabel='Smoothed reconstruction error', 
                  color_mean='navy', color_std='royalblue', color_min=None, plot_std=2, 
                  ax=None, pc_opt=None, title=None, label='', groupby=None, 
                  labelsize=labelsize, ticksize=ticksize, titlesize=titlesize, verbose=False,
                  add_fit=False, **kwargs):
    """
    Plot reconstruction error as a function of sampling probability (alternatively, plot results of any two parameters)
    :param L: dataframe with sampling parameters and errors
    :param pc_opt: add optimal pc to plot
    :param xcol: column name for x axis
    :param ycol: column name for y axis
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param color_mean: color for mean line
    :param color_std: color for standard deviation
    :param color_min: color for minimum error
    :param plot_std: number of standard deviations to plot
    :param ax: axis to plot on
    :param title: title for plot
    :param label: label for legend
    :param groupby: groupby to compute stats
    :param kwargs: additional arguments to pass to plt.plot"""


    ax = plt.subplots(figsize=(6, 6))[1] if ax is None else ax

    groupby = xcol if groupby is None else groupby
    L_grp = L.groupby(groupby)
    s_y = L_grp[ycol].std().values
    y = L_grp[ycol].mean().values
    x = L_grp[xcol].mean().values

    if 'linewidth' not in kwargs.keys():
        kwargs['linewidth'] = 3
    
    ax.plot(x, y, color=color_mean, label=label, **kwargs)
    for i in range(plot_std):
        v = (i + 1)
        ax.fill_between(x, np.array(y) + v * np.array(s_y), y, color=color_std, alpha=0.3/v)
        ax.fill_between(x, np.array(y) - v * np.array(s_y), y, color=color_std, alpha=0.3/v)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    title = title if title is not None else f'{ylabel} vs {xlabel}'
    ax.set_title(title, fontsize=titlesize)

    if pc_opt is not None:
        color_min = color_mean if color_min is None else color_min
        ax.axvline(x=pc_opt, color=color_min, linewidth=3, linestyle='--')

    ax.tick_params(axis='y', labelsize=ticksize)
    ax.tick_params(axis='x', labelsize=ticksize)
    ax.xaxis.label.set_size(labelsize)
    ax.yaxis.label.set_size(labelsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    # add linear fit
    if add_fit:
        model_ = linear_model.LinearRegression()
        model_.fit(L[[xcol]], L[ycol])
        if verbose:
            print(fr'{title}, coef: {model_.coef_}, int: {model_.intercept_}')
        ax.plot(L[xcol], model_.predict(L[[xcol]]), color='black', linewidth=4, linestyle='--')

        # add score
        r = 2 # R^2 round factor
        R = np.round(model_.score(x.reshape((-1,1)),y), r)
        kwargs_text = {'x': 0.3, 'y': 0.9, 'ha':'center', 'va':'center', 'fontsize':30}
        ax.text(s=fr'$R^2$={R}', transform=ax.transAxes, **kwargs_text)

        return model_


def plot_tradeoff_experiments(L_tradeoff, desc='', plot_std=0, plot_pc_opt=True, sharey=False, plot_pcs=True, xcol='pc', ycol='l1',
                            structures=[], colors={}, color='black', axs=None, title=None, **kwargs):
    """
    """
    # L_tradeoff['log pc'] = np.log(L_tradeoff['pc'])
    Bs = L_tradeoff['B'].unique()
    L_tradeoff_grp = L_tradeoff.groupby(['trajectory type'])
    structures = list(L_tradeoff_grp.groups.keys()) if structures == [] else structures
    colors = colors if colors != {} else {s: color for s in structures}
    
    nstructures = len(structures)
    nrepeats = np.max(L_tradeoff_grp['level_0'].nunique().values) if 'level_0' in L_tradeoff.columns else 1
    
    if 'level_0' not in L_tradeoff.columns:
        L_tradeoff.loc[:, 'level_0'] = 1

    if axs is None:
        if nrepeats == 1:
            fig, axs = plt.subplots(1, nstructures, figsize=(5*nstructures,5), sharex=True, sharey=sharey)
        else:
            fig, axs = plt.subplots(nstructures, nrepeats, figsize=(5*nrepeats, 5*nstructures), sharex=True, sharey=sharey)
            axs = axs.reshape((nstructures, nrepeats))

    
    istructure = 0
    for _,structure in enumerate(structures): # iterate over structures
        title = title if title is None else structure
    # for istructure,(structure,L) in enumerate(L_tradeoff_grp): # iterate over structures
        if structure in L_tradeoff_grp.groups.keys():
            L = L_tradeoff_grp.get_group(structure)
        else:
            continue
        color = colors[structure]
        for irepeat,(repeat,sL) in enumerate(L.groupby('level_0')): # iterate over repeats
            if isinstance(axs, np.ndarray):
                if len(axs.shape) == 1:
                    ax = axs[istructure]
                if len(axs.shape) == 2:
                    ax = axs[istructure, irepeat]
            else:
                ax = axs
            # if nrepeats == 1:
            #     if nstructures == 1:
            #         ax = axs
            #     else:
            #         ax = axs[istructure]
            # else:
            #     ax = axs[irepeat, istructure]
            
            for _,(B,ssL) in enumerate(sL.groupby('B')): # iterate over budgets
                pc_opt = find_min_nc(ssL, xcol=xcol, ycol=ycol) if plot_pc_opt else None
                plot_tradeoff(ssL, xcol=xcol, ycol=ycol, color_mean=color, color_std=color, 
                label=repeat, ax=ax, plot_std=plot_std, pc_opt=pc_opt, title=title, **kwargs)
        istructure += 1

    pcs = L_tradeoff['pc'].unique()
    text = f'B:{Bs}'
    if plot_pcs:
        ax.scatter(pcs, np.zeros_like(pcs) + L_tradeoff['l1'].min())
        ax.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center', 
        transform = ax.transAxes)
    # plt.suptitle(desc)
    plt.tight_layout()
    
    # footnote = f'B:{Bs}\\npc{pcs}'
    # plt.figtext(0.01, -0.3, footnote, fontsize=20)


def smooth_tradeoff(L, rollby='pc', roll=4):
    """
    Smooth out curve by averaging across consequent values
    :param L:
    :param rollby: roll by column
    :param roll: number of
    """
    # if there are multiple B
    Bs = L['B'].unique()
    L_per_B = []
    for B in Bs:
        sL = L[L['B'] == B]
        msL = sL.groupby([rollby]).mean()
        # msL['pc'] = msL.index
        L_per_B.append(msL.rolling(roll).mean().iloc[roll-1:].reset_index())
    rL = pd.concat(L_per_B)
    return rL


def check_increase_decrease(x, y, min_seq_locations=3, min_diff=1e-4):
    """
    Returns a vector specifying where function is decreasing or increasing (non-decreasing)
    """
    df = pd.DataFrame({'y':y, 'x': x})
    df = df.groupby('x').mean().reset_index()
    df.index = df['x']
    df.sort_index(inplace=True)
    
    df['dy'] = np.nan
    df['dy'].iloc[1:] = df.y.iloc[1:].values - df.y.iloc[:-1].values
    df = df.iloc[1:]
    # column for negative and positive
    df['sign'] = np.where(df['dy'] < 0, 'decreasing', 'increasing')
    df.loc[np.abs(df['dy']) < min_diff, 'sign'] = 'neither'
    # consecutive groups
    df['g'] = df['sign'].ne(df['sign'].shift()).cumsum()

    vals = df['g'].value_counts()
    vals = vals.index[vals.values >= min_seq_locations]
    vals = list(vals)
    return list(df[df['g'].isin(vals)]['sign'].unique())


def write_roman(num):
    """Convert number to roman index"""
    roman = OrderedDict()
    roman[1000] = "M"
    roman[900] = "CM"
    roman[500] = "D"
    roman[400] = "CD"
    roman[100] = "C"
    roman[90] = "XC"
    roman[50] = "L"
    roman[40] = "XL"
    roman[10] = "X"
    roman[9] = "IX"
    roman[5] = "V"
    roman[4] = "IV"
    roman[1] = "I"

    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= (r * x)
            if num <= 0:
                break

    return "".join([a for a in roman_num(num)])

    

def generate_gif(df, frameby='pc', xcol='PC1', ycol='PC2', fname='', 
                const_size=False, duration=20, unit="s", between="startend", **kwargs):
    """
    Generates gif of cell positions. 
    :param df: dataframe including frame id, and x and y position for each cell
    :param frameby: presents in a frame all cells (rows) with this id
    :param xcol: x position of each cell
    :param ycol: y position of each cell
    :param fname: name of gif file
    :param const_size: if True, all frames have the same limits
    """

    xlim = (df[xcol].min(), df[xcol].max())
    ylim = (df[ycol].min(), df[ycol].max())

    if not frameby.isin(df.columns):
        print(f'{frameby} not in columns. Exiting')
        return

    frameby_ids = df[frameby].unique().sort_values()
    nframes = len(frameby_ids)
    @gif.frame
    def plot(j):
        i = nframes - 1 - j
        frameby_val = frameby_ids[i]
        sdf = df[df[frameby] == frameby_val]
        plt.scatter(sdf[xcol], sdf[ycol], c= sdf['idx_c'], cmap='rainbow')
        
        plt.title(fr'{frameby}={np.round(frameby_val, 4)}')
        plt.axis('off')
        if const_size:
            plt.xlim(xlim)
            plt.ylim(ylim)
        
    frames = []
    for j in range(nframes):
        frame = plot(j)
        frames.append(frame)

    gif.save(frames, fname, duration=duration, unit=unit, between=between, **kwargs)