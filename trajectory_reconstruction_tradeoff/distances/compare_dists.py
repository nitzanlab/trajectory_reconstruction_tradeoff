import numpy as np
import netrd
import ot
import pandas as pd
from .compute_dists import graph_to_dists

import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def compare_distances(D0, D):
    """
    Compute error(deviation) between distances
    :param D0: true distances
    :param D: predicted distances
    :return:
        mean absolute error (l1)
        mean squared error (l2^2)
        mean squared error of log(1+x)
        mean correlation of distances
    """
    l1 = round(sum(sum(np.abs(np.array(D) - np.array(D0)))) / len(D) ** 2, 3)
    l2 = round(sum(sum(((np.array(D) - np.array(D0)) ** 2))) / len(D) ** 2, 3)
    l3 = round(sum(sum(((np.log(1 + np.array(D)) - np.log(1 + np.array(D0))) ** 2))) / len(D) ** 2, 3)  # rmlse

    lsp = []
    for r0, r in zip(D0, D):
        lsp.append(spearmanr(r0, r))
    lsp = np.mean(lsp)

    return l1, l2, l3, lsp




def ot_dist(G1, G2, p_labels, q_labels, plot=False):
    """

    :param G1:
    :param G2:
    :param p:
    :param q:
    """
    p = p_labels.value_counts()
    q = q_labels.value_counts()

    p_ord = list(p.index)
    q_ord = list(q.index)

    C1 = graph_to_dists(G1).loc[p_ord][p_ord].values
    C2 = graph_to_dists(G2).loc[q_ord][q_ord].values

    p /= p.sum()
    q /= q.sum()

    gw0, log0 = ot.gromov.gromov_wasserstein(C1, C2, p.values, q.values, 'square_loss', verbose=False, log=True)
    gw_dist = log0['gw_dist']

    if plot:
        plt.subplot(1, 2, 1);
        plt.imshow(C1);
        plt.subplot(1, 2, 2);
        plt.imshow(np.dot(gw0, np.dot(C2, gw0.T)));
        plt.show()
    return gw_dist


def him_dist(G1, G2):
    him = None
    if len(G1.nodes) == len(G2.nodes):
        dist_obj = netrd.distance.HammingIpsenMikhailov()
        him = dist_obj.dist(G1, G2)
    return him





