import numpy as np
import pandas as pd
from scipy.special import lambertw
epsilon = 10e-10

def softmax_max(xs, a=10):
    xs = np.array(xs)
    e_ax = np.exp(a * xs)
    return xs @ e_ax.T / np.sum(e_ax)









def find_min(x, y):
    """
    Find pc that minimizes the error
    :param x:
    :param y:
    :return:
    """
    return x[y == np.min(y)][0]

def find_min_nc(L, xcol='pc', ycol='l1'):
    """
    :param L: dataframe with sampling parameters and errors
    :return:
        optimal pc where error is minimal
    """
    L_by_xcol = L.groupby(xcol)[ycol]
    y = L_by_xcol.mean().values
    x = L_by_xcol.mean().index.values

    pc_opt = find_min(x, y)

    return pc_opt


def infer_complex_nr(nc1, nc2, B1, B2):
    """
    Infer nr from pairs of (budget, optimal cell capture)
    :return:
    """
    alpha = nc2 / nc1
    K = B1 / B2
    nr = np.power(K * alpha, (1/(1 - alpha)))
    return nr

def swap(a,b):
    return b,a

def infer_optimal_nc(B, B1, nc1, B2, nc2, traj_type='simple'):
    """
    Infer the optimal nc for a new experiment
    :param B: new budget
    :param B1:
    :param nc1:
    :param B2:
    :param nc2:
    :param traj_type: 'simple' or 'complex' depending on curvatures and bifurcations
    :return:
    """
    if traj_type == 'simple':
        return np.mean((nc1, nc2))
    if traj_type == 'complex':
        if (B1 > B2) and (nc1 > nc2): # swap
            B1, B2 = swap(B1, B2)
            nc1, nc2 = swap(nc1, nc2)
        if not ((B2 > B1) and (nc2 > nc1)):
            print('For complex trajectory expecting nc to increase with budgets increase')
            return
        nr1 = infer_complex_nr(nc1, nc2, B1, B2)
        nc = nc1 * lambertw(B/B1 * nr1 * np.log(nr1)) / np.log(nr1)
        return nc

if __name__ == '__main__':
    B1 = 0.00002
    B2 = 0.0002
    nc1 = 0.25
    nc2 = 0.28
    B = np.linspace(B1, B2, 30)
    nc = infer_optimal_nc(B, B1, nc1, B2, nc2, traj_type='complex')
    import matplotlib.pyplot as plt
    plt.scatter(B, nc, s=100)
    plt.scatter(B1, nc1, c='r', s=100)
    plt.scatter(B2, nc2, c='r', s=100)
    plt.show()
