
import numpy as np
import pandas as pd
import altair as alt
import matplotlib
from collections import OrderedDict
import matplotlib.pyplot as plt
from trajectory_reconstruction_tradeoff.opt import find_min_nc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import EllipseCollection
import seaborn as sns

plt.rcParams.update({'figure.max_open_warning': 0})
titlesize = 35
labelsize = 30
ticksize = 25
legendsize = 25

# color_map = {'A': '#e6194b', 'B': '#3cb44b', 'C': '#ffe119', 'D': '#4363d8', 'E': '#f58231', 'F': '#911eb4',
#              'G': '#46f0f0', 'H': '#f032e6', 'I': '#bcf60c', 'J': '#fabebe', 'K': '#008080', 'L': '#e6beff',
#              'M': '#9a6324', 'N': '#fffac8', 'O': '#800000', 'P': '#aaffc3', 'Q': '#808000', 'R': '#ffd8b1',
#              'S': '#000075', 'T': '#808080'}


# def plot_pca2d(expr_red, meta=None, color=None, sigma_expr=None, color_sigma='b', title='', fname=None, ax=None,
#                color_type='N', pt_size=60, cmap='cold', shade=None, shade_shift=0.001):
#     """
#     Plot expression in reduced space
#     :param expr_red: expression reduced representation
#     :param sigma_expr: noise around reduced representation
#     """
#     ax_none = ax is None
#     if ax_none:
#         fig, ax = plt.subplots(figsize=(10, 7))

#     color_cell = 'k'
#     # if meta is not None and color is not None:
#     #     color_cell = meta[color]
#     #     color_cell = list(map(color_map.get, color_cell)) if color == 'branch' else color_cell
#     if meta is not None and color is not None:
#         color_cell = meta[color]
#         color_cell_un = color_cell.unique()
#         if not isinstance(color_cell.dtype, float) and (color_type == 'N'):
#             cmap = plt.cm.get_cmap('tab20')
#             cmap_hex = {ce: matplotlib.colors.rgb2hex(cmap.colors[np.mod(i, 20)]) for i, ce in enumerate(color_cell_un)}
#             color_cell = list(map(cmap_hex.get, color_cell))

#     if shade is not None:
#         shade_shift = shade_shift * expr_red[:, [0,1]].max()
#         ax.scatter(expr_red[:, 0] - shade_shift, expr_red[:, 1] - shade_shift, c=shade, marker='.', s=pt_size)
#     ax.scatter(expr_red[:, 0], expr_red[:, 1], c=color_cell, marker='.', s=pt_size)
#     # color_sigma = color_sigma if branch is None else list(map(color_map.get, branch))
#     if sigma_expr is not None:
#         ax.add_collection(EllipseCollection(widths=sigma_expr, heights=sigma_expr, angles=0, units='xy',
#                                             offsets=list(zip(expr_red[:, 0], expr_red[:, 1])), transOffset=ax.transData,
#                                             alpha=0.4, color=color_sigma))
#     ax.set_title(title)
#     ax.set_xlabel('PC0')
#     ax.set_ylabel('PC1')
#     # plt.colorbar()
#     if fname is not None:
#         plt.savefig(fname)
#     elif ax_none:
#         plt.show()

def plot_pca2d(pX, meta=None, color='', title='', fname=None, ax=None,
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

    colorlabel = color.title() if colorlabel is None else colorlabel
    

    if (meta is not None) and (color in meta.columns):
        df[colorlabel] = meta[color].values
        kwargs['hue'] = colorlabel if 'hue' not in kwargs.keys() else kwargs['hue']

    sns.scatterplot(data=df, x=xlabel, y=ylabel, ax=ax, **kwargs)
    ax.set_title(title, fontsize=titlesize)
    ax.axis('off')
    if legend:
        ax.legend(fontsize=legendsize)
        # Add a legend
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
        ax.legend(handletextpad=0.01,
                  loc='lower center', 
                  bbox_to_anchor=(0.5, -0.5),
                  ncol=2, fontsize=20, frameon=False)
    else:
        ax.get_legend().remove()

    
    if fname is not None:
        plt.savefig(fname)
    elif ax_none:
        plt.show()



def plot_tradeoff(L, xcol='pc', ycol='l1', xlabel='Sampling probability', ylabel='Smoothed reconstruction error', 
                  color_mean='navy', color_std='royalblue', color_min=None, plot_std=2, 
                  ax=None, pc_opt=None, title=None, label='', groupby=None, 
                  labelsize=labelsize, ticksize=ticksize, titlesize=titlesize, **kwargs):
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
    if title is not None:
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

# def to_paper(pl):
#     labelFontSize=15
#     titleFontSize=20
#     fontSize=20

#     pl = pl.configure_view(strokeOpacity=1, strokeWidth=3, stroke='black')
#     pl = pl.configure_axis(labelFontSize=labelFontSize, titleFontWeight='normal', titleFontSize=titleFontSize)
#     pl = pl.configure_title(fontSize=fontSize)
#     pl = pl.configure_legend(titleFontSize=labelFontSize, labelFontSize=labelFontSize)

#     return pl

# def plot_tradeoff_dw(res, Pcs, ycol, ylabel):
#     axis_nada = alt.Axis(grid=False, labels=False, ticks=False)
#     scale_nz = alt.Scale(zero=False)

#     color_var = '#b1b1b1'
#     color_few_cells = '#df755b'
#     color_med_cells = '#5bb844'
#     color_lots_cells = '#7997dc'

#     back_color_truth = '#d8d8d8'
#     back_color_few_cells = '#fde7e2'
#     back_color_med_cells = '#e2f7df'
#     back_color_lots_cells = '#eaf0fc'

#     colors = [color_few_cells, color_med_cells, color_lots_cells, ]
#     back_colors = [back_color_few_cells, back_color_med_cells, back_color_lots_cells, back_color_truth]

#     xlabel = 'Cell sampling probability (pc)'
#     x = alt.X('pc:O', #scale=alt.Scale(type='log', base=1.000001),
#               title=xlabel,)# axis=alt.Axis(grid=False, values=np.linspace(0,1, 6)))
# #     yticks = np.linspace(res[ycol].min(), res[ycol].max(), 6)
#     y = alt.Y(ycol + ':Q', axis=alt.Axis(grid=False, tickCount=6), scale=scale_nz, title=ylabel)
#     pl_all = alt.Chart(res, width=300, height=300).mark_boxplot().encode(x=x,
#                                                                          y=y,
#                                                                          color=alt.value(color_var))

#     pls = [pl_all]

#     for i, nc in enumerate(Pcs):

#     #     print(df[sel2]['pc'].unique())
#         pls.append(alt.Chart(res[res['pc'] == nc]).mark_boxplot().encode(x=x,
#                                                                  y=y,
#                                                                  color=alt.value(colors[i])))

#     return to_paper(alt.layer(*pls))




def plot_tradeoff_experiments(L_tradeoff, desc='', plot_std=0, plot_pc_opt=True, sharey=False, plot_pcs=True, xcol='pc', ycol='l1',
                            structures=[], colors={}, color='black', axs=None, **kwargs):

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
                label=repeat, ax=ax, plot_std=plot_std, pc_opt=pc_opt, title=structure, **kwargs)
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