
import numpy as np
import pandas as pd
import altair as alt
import gif
from collections import OrderedDict
import matplotlib.pyplot as plt
from trajectory_reconstruction_tradeoff.opt import find_min_nc
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from .plotting_configs import get_color_col
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from .saturation_model import SaturationModel
from mycolorpy import colorlist as mcp

plt.rcParams.update({'figure.max_open_warning': 0})
titlesize = 35
labelsize = 30
ticksize = 25
legendsize = 28

# regression models
models = {'linear': LinearRegression,
          'huber': HuberRegressor,
          'saturation': SaturationModel,
          }

def to_paper(pl):
    """
    Beautify altair chart
    :param pl: altair chart 
    """
    labelFontSize=15
    titleFontSize=20
    fontSize=20

    pl = pl.configure_view(strokeOpacity=0)
    pl = pl.configure_axis(labelFontSize=labelFontSize, titleFontWeight='normal', titleFontSize=titleFontSize)
    pl = pl.configure_title(fontSize=fontSize)
    pl = pl.configure_legend(titleFontSize=labelFontSize, labelFontSize=labelFontSize)

    return pl
    
def plot_3d(pX, meta=None, color=None, title='', fname=None, ax=None,
               xlabel = 'PC1', ylabel = 'PC2', colorlabel=None, legend=True, legendsize=legendsize, titlesize=titlesize, palette=None, dazim=90, delev=-20, **kwargs):
    """
    Plot 3d 
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
        fig = plt.figure(figsize=(6,6))
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

    
    pX = pX.values if isinstance(pX, pd.DataFrame) else pX
    
    df = pd.DataFrame({xlabel: pX[:,0], ylabel: pX[:,1]})

    color = get_color_col(meta, color=color)
    colorlabel = color.title() if (colorlabel is None) and color else colorlabel

    if (meta is not None) and color and palette:
        df[colorlabel] = meta[color].values
        kwargs['c'] = df[colorlabel].map(palette)

    # plot
    ax.scatter(pX[:,0], pX[:,1], pX[:,2], **kwargs)
    ax.set_title(title, fontsize=titlesize)
    

    # Add a legend
    if color and legend:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
        ax.legend(handletextpad=0.01,
                  loc='lower center', 
                  bbox_to_anchor=(0.5, -0.5),
                  ncol=2, fontsize=legendsize, frameon=False)
    elif color and not legend:
        ax.get_legend().remove()

    ax.view_init(ax.elev + delev, ax.azim + dazim)
    if fname is not None:
        plt.savefig(fname)
    elif ax_none:
        plt.show()



def plot_2d(pX, meta=None, color=None, title='', fname=None, ax=None,
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

    color = get_color_col(meta, color=color)
    colorlabel = color.title() if (colorlabel is None) and color else colorlabel
    

    if (meta is not None) and (color):
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

def project(pX, n_comp=2):
    """
    Project onto PCA space
    :param pX: cells reduced representation
    """
    pca = PCA(n_components=n_comp, svd_solver='full')
    ppX = pca.fit_transform(pX)
    pcnames = ['PC%d' % (i+1) for i in np.arange(ppX.shape[1])]
    ppX = pd.DataFrame(ppX, index=pX.index, columns=pcnames)
    return ppX

def plot_project_pca2d(pX, **kwargs):
    """
    Projects and plots to 2d PCA space
    :param pX: cells reduced representation
    """
    ppX = project(pX, n_comp=2)
    plot_2d(ppX, **kwargs)


def plot_spring_layout(pX=None, A=None, neighbors=2, **kwargs):
    """
    Plots spring layout of the minimal-fully-connected kNN graph
    :param pX: cells reduced representation
    :param neighbors: number of neighbors
    """
    # distances are not given and have to be recomputed
    if A is None and pX is not None:
        A = kneighbors_graph(pX, neighbors, mode='distance', metric='euclidean', include_self=False)
    
    G = nx.from_numpy_matrix(A)

    # set position by pX
    if pX is not None:
        pos = {}
        for inode,node in enumerate(G.nodes()):
            pos[node] = pX.iloc[inode].values[:2]
    else:
        pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=3, **kwargs) 
    plt.show()


def plot_tradeoff(L, xcol='pc', ycol='l1', xlabel=None, ylabel=None, 
                  color_mean='slategray', color_std='slategray', color_min=None, plot_std=1, xcol_twin=None, twin_values=None,
                  ax=None, pc_opt=None, title=None, label='', groupby=None, 
                  labelsize=labelsize, ticksize=ticksize, titlesize=titlesize, verbose=False,
                  add_fit=False, add_R=False, model_type='huber', **kwargs):
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

    xlabel = xcol if xlabel is None else xlabel
    ylabel = ycol if ylabel is None else ylabel

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

    if xcol_twin is not None and xcol_twin in L.columns:
        ax_twin = ax.twiny()
        x2 = L_grp[xcol_twin].mean().values
        n = len(x2)
        n_ticks = 4

        min_xcol_twin = x2.min()
        max_xcol_twin = x2.max()
        x2_values = [t for t in twin_values if (t > min_xcol_twin and t < max_xcol_twin)]
            
        if x2_values is not None and len(x2_values) > 0:
            pts = pd.Series(x, index=x2)
            new_pts = pd.Series(np.nan, index=x2_values)
            x2_x = pd.concat((pts, new_pts))
            x2_x = x2_x.interpolate(method='index').groupby(level=0).mean()
            new_tick_locations = x2_x.loc[x2_values].values
            new_tick_values = x2_values
        else:
            idx = [int(n*i/(n_ticks+1)) for i in np.arange(1, n_ticks)]
            new_tick_locations = x[idx]
            new_tick_values = x2[idx].astype(int)

        ax_twin.set_xlim(ax.get_xlim())
        ax_twin.set_xticks(new_tick_locations)
        ax_twin.set_xticklabels(new_tick_values) 
        ax_twin.tick_params(axis='x', labelsize=ticksize)
        ax_twin.xaxis.label.set_size(labelsize)

    # add linear fit
    if add_fit:
        model_ = models[model_type]()
        model_.fit(L[[xcol]], L[ycol])
        if verbose:
            if model_.x0_ is None:
                print(fr'{title}, coef: {model_.coef_}, int: {model_.intercept_}')
            else:
                print(fr'{title}, coef: {model_.coef_}, int: {model_.intercept_}, saturation: {model_.x0_}')
        x_new = np.linspace(L[xcol].min(), L[xcol].max(), 10)
        ax.plot(x_new, model_.predict(x_new.reshape(-1,1)), color='black', linewidth=4, linestyle='--')

        # add score
        r = 2 #  round factor
        R = np.round(model_.score(L[xcol].values.reshape((-1, 1)), L[ycol].values.reshape((-1, 1))), r) # assuming a single group
        kwargs_text = {'x': 0.3, 'y': 0.9, 'ha':'center', 'va':'center', 'fontsize':30}
        if add_R:
            ax.text(s=fr'$R^2$={R}', transform=ax.transAxes, **kwargs_text)

        return model_


def plot_tradeoff_experiments(L_tradeoff, plot_pc_opt=True, sharey=False, plot_pcs=True, xcol='pc', ycol='l1',
                            structures=[], color_groupby='trajectory type', colors={}, color='black', axs=None, title=None, **kwargs):
    """
    Plot tradeoff experiments
    :param L_tradeoff: pandas dataframe with columns: pc, l1, B, trajectory type, level_0
    :param desc: description of the experiment
    :param plot_std: number of standard deviations to plot
    :param plot_pc_opt: plot pc_opt
    :param sharey: share y axis
    :param plot_pcs: plot pcs
    :param xcol: x column
    :param ycol: y column
    :param structures: list of structures to plot
    :param color_groupby: column to group by
    :param colors: dictionary of colors
    """
    Bs = L_tradeoff['B'].unique()
    L_tradeoff_grp = L_tradeoff.groupby([color_groupby])
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
            
            for _,(B,ssL) in enumerate(sL.groupby('B')): # iterate over budgets
                pc_opt = find_min_nc(ssL, xcol=xcol, ycol=ycol) if plot_pc_opt else None
                plot_tradeoff(ssL, xcol=xcol, ycol=ycol, color_mean=color, color_std=color, 
                label=repeat, ax=ax, pc_opt=pc_opt, title=title, **kwargs)
        istructure += 1

    pcs = L_tradeoff[xcol].unique()
    text = f'B:{Bs}'
    if plot_pcs:
        ax.scatter(pcs, np.zeros_like(pcs) + L_tradeoff['l1'].min())
        ax.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center', 
        transform = ax.transAxes)
    plt.tight_layout()


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

def plot_tradeoff_dw(res, Pcs, ycol, ylabel):
    """
    Plot downstream expression tradeoff
    :param res: dataframe with columns: pc, l1, B, trajectory type, level_0
    :param Pcs: list of PCs
    :param ycol: column name for y axis
    :param ylabel: label for y axis
    """
    axis_nada = alt.Axis(grid=False, labels=False, ticks=False)
    scale_nz = alt.Scale(zero=False)

    color_var = '#b1b1b1'
    color_few_cells = '#df755b'
    color_med_cells = '#5bb844'
    color_lots_cells = '#7997dc'

    back_color_truth = '#d8d8d8'
    back_color_few_cells = '#fde7e2'
    back_color_med_cells = '#e2f7df'
    back_color_lots_cells = '#eaf0fc'

    colors = [color_few_cells, color_med_cells, color_lots_cells, ]
    back_colors = [back_color_few_cells, back_color_med_cells, back_color_lots_cells, back_color_truth]

    xlabel = ''
    x = alt.X('pc:O', title=xlabel,)
    y = alt.Y(ycol + ':Q', axis=alt.Axis(grid=False, tickCount=6), scale=scale_nz, title=ylabel)
    pl_all = alt.Chart(res, width=300, height=300).mark_boxplot(outliers=False).encode(x=x, y=y, color=alt.value(color_var))
    pls = [pl_all]

    for i, nc in enumerate(Pcs):
        pls.append(alt.Chart(res[res['pc'] == nc]).mark_boxplot(outliers=False).encode(x=x,
                                                                 y=y,
                                                                 color=alt.value(colors[i])))

    return to_paper(alt.layer(*pls))
    

def get_colors_by_budget(Bs, cmap="PRGn"):
    """
    A colormap for budget values
    :param Bs: list of budgets
    """
    nBs = len(Bs)
    colors_for_budget = mcp.gen_color(cmap=cmap, n=nBs+5)
    colors_for_budget = colors_for_budget[:int(np.ceil(nBs/2))] + colors_for_budget[-int(np.floor(nBs/2)):]
    colors_by_budget = {B: colors_for_budget[i] for i, B in enumerate(Bs)}
    return colors_by_budget
