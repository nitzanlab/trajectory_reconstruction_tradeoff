import numpy as np
import pandas as pd
from scipy.special import lambertw
from scipy.optimize import curve_fit
from ..plotting.saturation_model import SaturationModel
epsilon = 10e-10

# def softmax_max(xs, a=10):
#     """
#     Computes softmax of an array
#     """
#     xs = np.array(xs)
#     e_ax = np.exp(a * xs)
#     return xs @ (e_ax.T / np.sum(e_ax, axis=1))
def softmax_max(xs, a=1):
    """
    Computes softmax of an array
    """
    xs = np.array(xs).T
    e_ax = np.exp(a * xs)
    return (xs * (e_ax.T / np.sum(e_ax, axis=1)).T).sum(axis=1)



def get_quadratic_sol(a,b,c):
    """
    Given quadratic formula of the form: ax^2 + bx + c = 0
    solves for its roots.
    """
    sol1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2 * a)
    sol2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2 * a)
    return sol1,sol2


def get_pc_min_pred_inv_sqrt(model_read, model_cell, B):
    """
    Given the linear fits of:
    :param model_read: read downsample error as a function of 1/sqrt(pt)
    :param model_cell: cell downsample error as a function of 1/sqrt(pc)
    :param B: subsampling budget
    computes the optimal number of cells to assay
    """
    a = model_read.intercept_.take(0)
    b = model_read.coef_.take(0)
    alpha = model_cell.intercept_.take(0)
    beta = model_cell.coef_.take(0)

    # TODO: check both errors are increasing/decreasing as should
    if (a < 0) or (alpha < 0):
        print(f'Intercepts should be non-negative. Got, for read model: {a}, for cell model: {alpha}')
        return

    if (b < 0) or (beta < 0):
        print(f'Slopes should be non-negative. Got, for read model: {b}, for cell model: {beta}')
        return

    w = (alpha-a)
    v = -b / np.sqrt(B)
    t = beta

    sol1, sol2 = get_quadratic_sol(v,w,t)

    pc_min_pred = sol2**2
    return pc_min_pred


def get_pc_min_pred_log(m1, m2, B):
    """
    Given two linear fits, m1 and m2, describing functions f1(B/x) and f2(x) respectively, finds for which x, f1(B/x) = f2(x)
    :param m1: read downsample error as a function of 1/sqrt(pt)
    :param m2: cell downsample error as a function of 1/sqrt(pc)
    :param B: subsampling budget
    computes the optimal number of cells to assay
    """
    a = m1.intercept_.take(0)
    b = m1.coef_.take(0)
    alpha = m2.intercept_.take(0)
    beta = m2.coef_.take(0)

    pc_hat = np.exp((b*np.log(B) + a - alpha) / (beta + b))
    pc_hat = np.array(pc_hat)
    # print(pc_hat)
    if isinstance(m1, SaturationModel):
        pt_hat = B / pc_hat
        idx_sat = np.log(pt_hat) > m1.x0_
        pc_hat[idx_sat] = B[idx_sat] / np.exp(m1.x0_) #np.exp((m1.y0_ - alpha)/beta)
    return pc_hat

def get_pc_min_pred_log_log(m1, m2, B):
    """
    Given two linear fits, m1 and m2, describing functions f1(B/x) and f2(x) respectively, finds for which x, f1(B/x) = f2(x)
    :param m1: read downsample error as a function of 1/sqrt(pt)
    :param m2: cell downsample error as a function of 1/sqrt(pc)
    :param B: subsampling budget
    computes the optimal number of cells to assay
    """
    a = m1.intercept_.take(0)
    b = m1.coef_.take(0)
    alpha = m2.intercept_.take(0)
    beta = m2.coef_.take(0)

    xhat = np.exp((b*np.log(B) + a - alpha) / (beta + b))

    return xhat


# def fit_reconstruction_err(L):
#     """
#     Fit each reconstruction error curve by:
#     \varepsilon = b * (sqrt(B)/x) + beta * x +c
#     where x corresponds to 1/sqrt(pc)
#     """
#     # TODO: check for single experiment (budget and dataset)
#     nBs = L['B'].value_counts().shape[0]
#     ntrajs = L['trajectory type'].value_counts().shape[0] if 'trajectory type' in L.columns else 0
#     if (nBs > 1) or (ntrajs > 1):
#         print(f'Computes fit of a single experiment (budget and data). Data seems to include: {ntrajs} data types, and {nBs} budgets')
    
#     gL = L.groupby('pc').mean()
#     sqB = np.sqrt(L['B'].values[0])

#     def cov_err(x, b, beta, c):
#         y = b*(sqB/x) + beta*x + c
#         return y

#     xdata = gL['sqrt inv pc'].values
#     ydata = gL['l1'].values
#     parameters, _ = curve_fit(cov_err, xdata, ydata)

#     ydata_hat = cov_err(xdata, *parameters)
    
#     # plt.plot(xdata, ydata, 'o', label='data')
#     # plt.plot(xdata, ydata_hat, '-', label='fit')
    
#     return ydata_hat
    

# def fit_reconstruction_err(L):
#     """
#     Fit each reconstruction error curve by:
#     \varepsilon = b * (x2/x1) + beta * x1 +c
#     where x1 corresponds to 1/sqrt(pc) and x2 corresponds to sqrt(B)
#     """
#     ntrajs = L['trajectory type'].value_counts().shape[0] if 'trajectory type' in L.columns else 0
#     if (ntrajs > 1):
#         print(f'Computes fit of a single trajectory. Data seems to include: {ntrajs} trajectories.')
    
#     # gL = L.groupby(['B','pc']).mean().reset_index()
    
#     # def cov_err(X, b, beta, c):
#     #     x1,x2 = X
#     #     y = b*(x2/x1) + beta*x1 + c
#     #     return y

#     def cov_err(X, b, beta, a, alpha):
#         x1,x2 = X
#         y = np.max(b*(x2/x1) + a, beta*x1 + alpha)
#         return y

#     xdata1 = L['sqrt inv pc'].values
#     xdata2 = np.sqrt(L['B'].values)
#     xdata = (xdata1, xdata2)
#     ydata = L['l1'].values
#     parameters, _ = curve_fit(cov_err, xdata, ydata)

#     ydata_hat = cov_err(xdata, *parameters)
    
#     # plt.plot(xdata, ydata, 'o', label='data')
#     # plt.plot(xdata, ydata_hat, '-', label='fit')
    
#     L['pred l1 fit'] = ydata_hat
#     return L

def fit_reconstruction_err(xdata1, xdata2, ydata, get_params=False):
    """
    Fit reconstruction error curves by:
    ydata = max(alpha + beta * xdata1, a + b * xdata2)
    """
    
    def cov_err(X, b, beta, a, alpha):
        x1,x2 = X
        # w = x1 / (x1 + x2)
        # y = w*(b*x2 + a) + (1-w)*(beta*x1 + alpha)
        # y = softmax_max([b*x2 + a, beta*x1 + alpha])
        y = np.maximum(b*x2 + a, beta*x1 + alpha)
        return y

    xdata = (xdata1, xdata2)
    parameters, _ = curve_fit(cov_err, xdata, ydata)

    ydata_hat = cov_err(xdata, *parameters)

    if get_params:
        return ydata_hat, *parameters
    else:
        return ydata_hat



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
