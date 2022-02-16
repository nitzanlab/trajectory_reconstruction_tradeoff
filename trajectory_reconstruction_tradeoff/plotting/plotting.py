
import numpy as np
import altair as alt
import matplotlib
import matplotlib.pyplot as plt
from utils import find_min
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import EllipseCollection

plt.rcParams.update({'figure.max_open_warning': 0})

color_map = {'A': '#e6194b', 'B': '#3cb44b', 'C': '#ffe119', 'D': '#4363d8', 'E': '#f58231', 'F': '#911eb4',
             'G': '#46f0f0', 'H': '#f032e6', 'I': '#bcf60c', 'J': '#fabebe', 'K': '#008080', 'L': '#e6beff',
             'M': '#9a6324', 'N': '#fffac8', 'O': '#800000', 'P': '#aaffc3', 'Q': '#808000', 'R': '#ffd8b1',
             'S': '#000075', 'T': '#808080'}


def plot_pca2d(expr_red, meta=None, color=None, sigma_expr=None, color_sigma='b', title='', fname=None, ax=None,
               color_type='N', pt_size=60):
    """
    Plot expression in reduced space
    :param expr_red: expression reduced representation
    :param sigma_expr: noise around reduced representation
    """
    ax_none = ax is None
    if ax_none:
        fig, ax = plt.subplots(figsize=(10, 7))

    color_cell = 'k'
    # if meta is not None and color is not None:
    #     color_cell = meta[color]
    #     color_cell = list(map(color_map.get, color_cell)) if color == 'branch' else color_cell
    if meta is not None and color is not None:
        color_cell = meta[color]
        color_cell_un = color_cell.unique()
        if not isinstance(color_cell.dtype, float) and (color_type == 'N'):
            cmap = plt.cm.get_cmap('tab20')
            cmap_hex = {ce: matplotlib.colors.rgb2hex(cmap.colors[np.mod(i, 20)]) for i, ce in enumerate(color_cell_un)}
            color_cell = list(map(cmap_hex.get, color_cell))

    ax.scatter(expr_red[:, 0], expr_red[:, 1], c=color_cell, marker='.', s=pt_size)
    # color_sigma = color_sigma if branch is None else list(map(color_map.get, branch))
    if sigma_expr is not None:
        ax.add_collection(EllipseCollection(widths=sigma_expr, heights=sigma_expr, angles=0, units='xy',
                                            offsets=list(zip(expr_red[:, 0], expr_red[:, 1])), transOffset=ax.transData,
                                            alpha=0.4, color=color_sigma))
    ax.set_title(title)
    ax.set_xlabel('PC0')
    ax.set_ylabel('PC1')
    # plt.colorbar()
    if fname is not None:
        plt.savefig(fname)
    elif ax_none:
        plt.show()


def plot_tradeoff(L, xcol='pc', ycol='l1', xlabel='Sampling probability',
                  color_mean='navy', color_std='royalblue', color_min=None,
                  ax=None, pc_opt=None):
    """
    Plot opt - reconstruction error as the opt bw pc and pt shifts under constant budget
    :param L: dataframe with sampling parameters and errors
    :param pc_opt: add optimal pc to plot
    :return:
        optimal pc where error is minimal
    """
    ax = plt.subplots(figsize=(14, 10))[1] if ax is None else ax

    L_by_xcol = L.groupby(xcol)[ycol]
    s_y = L_by_xcol.std().values
    y = L_by_xcol.mean().values
    x = L_by_xcol.mean().index.values

    ax.plot(x, y, color=color_mean, linewidth=3)
    ax.fill_between(x, np.array(y) + np.array(s_y), y, color=color_std, alpha=0.3)
    ax.fill_between(x, np.array(y) - np.array(s_y), y, color=color_std, alpha=0.3)
    ax.fill_between(x, np.array(y) + 2 * np.array(s_y), y, color=color_std, alpha=0.15)
    ax.fill_between(x, np.array(y) - 2 * np.array(s_y), y, color=color_std, alpha=0.15)

    ax.set_xlabel(xlabel);
    ax.set_ylabel('Smoothed reconstruction error');

    if pc_opt is not None:
        color_min = color_mean if color_min is None else color_min
        ax.axvline(x=pc_opt, color=color_min, linewidth=3, linestyle='--')


def to_paper(pl):
    labelFontSize=15
    titleFontSize=20
    fontSize=20

    pl = pl.configure_view(strokeOpacity=1, strokeWidth=3, stroke='black')
    pl = pl.configure_axis(labelFontSize=labelFontSize, titleFontWeight='normal', titleFontSize=titleFontSize)
    pl = pl.configure_title(fontSize=fontSize)
    pl = pl.configure_legend(titleFontSize=labelFontSize, labelFontSize=labelFontSize)

    return pl

def plot_tradeoff_dw(res, Pcs, ycol, ylabel):
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

    xlabel = 'Cell sampling probability (pc)'
    x = alt.X('pc:O', #scale=alt.Scale(type='log', base=1.000001),
              title=xlabel,)# axis=alt.Axis(grid=False, values=np.linspace(0,1, 6)))
#     yticks = np.linspace(res[ycol].min(), res[ycol].max(), 6)
    y = alt.Y(ycol + ':Q', axis=alt.Axis(grid=False, tickCount=6), scale=scale_nz, title=ylabel)
    pl_all = alt.Chart(res, width=300, height=300).mark_boxplot().encode(x=x,
                                                                         y=y,
                                                                         color=alt.value(color_var))


    pls = [pl_all]

    for i, nc in enumerate(Pcs):

    #     print(df[sel2]['pc'].unique())
        pls.append(alt.Chart(res[res['pc'] == nc]).mark_boxplot().encode(x=x,
                                                                 y=y,
                                                                 color=alt.value(colors[i])))

    return to_paper(alt.layer(*pls))
