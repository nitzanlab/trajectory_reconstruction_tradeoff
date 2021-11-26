import matplotlib.pyplot as plt
import numpy as np
from utils import find_min
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import EllipseCollection

plt.rcParams.update({'figure.max_open_warning': 0})

color_map = {'A': '#e6194b', 'B': '#3cb44b', 'C': '#ffe119', 'D': '#4363d8', 'E': '#f58231', 'F': '#911eb4',
             'G': '#46f0f0', 'H': '#f032e6', 'I': '#bcf60c', 'J': '#fabebe', 'K': '#008080', 'L': '#e6beff',
             'M': '#9a6324', 'N': '#fffac8', 'O': '#800000', 'P': '#aaffc3', 'Q': '#808000', 'R': '#ffd8b1',
             'S': '#000075', 'T': '#808080'}

def plot_pca2d(expr_red, sigma_expr=None, branch=None, color_sigma='b', title='', fname=None):
    """
    Plot expression in reduced space
    :param expr_red: expression reduced representation
    :param sigma_expr: noise around reduced representation
    """
    fig, ax = plt.subplots(figsize=(10,7))
    scatter = plt.scatter(expr_red[:, 0], expr_red[:, 1],  color='k', marker='.', s=60)
    color_sigma = color_sigma if branch is None else list(map(color_map.get, branch))
    if sigma_expr is not None:
        ax.add_collection(EllipseCollection(widths=sigma_expr, heights=sigma_expr, angles=0, units='xy',
                                            offsets=list(zip(expr_red[:, 0], expr_red[:, 1])), transOffset=ax.transData,
                                            alpha=0.4, color=color_sigma))
    plt.title(title)
    plt.xlabel('PC0')
    plt.ylabel('PC1')
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()


def find_min(x, y):
    """
    Find pc that minimizes the error
    :param x:
    :param y:
    :return:
    """
    return x[y == np.min(y)]

def plot_tradeoff(L, xcol='pc', ycol='l1', xlabel='Sampling probability',
                  color_mean='navy', color_std='royalblue', color_min=None,
                  plot_min=True, ax=None):
    """
    Plot tradeoff - reconstruction error as the tradeoff bw pc and pt shifts under constant budget
    :param L: dataframe with sampling parameters and errors
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

    pc_opt = find_min(x, y)
    if plot_min:
        color_min = color_mean if color_min is None else color_min
        ax.axvline(x=pc_opt, color=color_min, linewidth=3, linestyle='--')

    return pc_opt
