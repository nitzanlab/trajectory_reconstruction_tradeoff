import numpy as np
import pandas as pd
from scipy.special import lambertw
from scipy.optimize import curve_fit
from ..plotting.saturation_model import SaturationModel
epsilon = 10e-10


def softmax_max(xs, a=1):
    """
    Computes softmax of an array
    """
    xs = np.array(xs)
    e_ax = np.exp(a * xs)
    return (xs * (e_ax.T / np.sum(e_ax, axis=1)).T).sum(axis=1)


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
    if isinstance(m1, SaturationModel):
        pt_hat = B / pc_hat
        idx_sat = np.log(pt_hat) > m1.x0_
        pc_hat[idx_sat] = B[idx_sat] / np.exp(m1.x0_) 
    return pc_hat


def fit_reconstruction_err(xdata1, xdata2, ydata, get_params=False):
    """
    Fit reconstruction error curves by:
    ydata = max(alpha + beta * xdata1, a + b * xdata2)
    """
    
    def cov_err(X, b, beta, a, alpha):
        x1,x2 = X
        y = np.maximum(b*x2 + a, beta*x1 + alpha)
        return y

    xdata = (xdata1, xdata2)
    parameters, _ = curve_fit(cov_err, xdata, ydata)

    ydata_hat = cov_err(xdata, *parameters)

    if get_params:
        return ydata_hat, *parameters
    else:
        return ydata_hat
    

def find_min_nc(L, xcol='pc', ycol='l1'):
    """
    :param L: dataframe with sampling parameters and errors
    :return:
        optimal pc where error is minimal
    """
    L_by_xcol = L.groupby(xcol)[ycol]
    y = L_by_xcol.mean().values
    x = L_by_xcol.mean().index.values

    pc_opt = x[y == np.min(y)][0] # finds minimal error

    return pc_opt


def continuously_in_range(df, xcol, ycol, xval, yval, yrange, below=True):
    """
    Checks if a point is within a range of another point
    """
    df_bel = df.loc[df[xcol] < xval] if below else df.loc[df[xcol] > xval]
    df_bel['dist to opt'] = np.abs(df_bel[xcol] - xval)
    df_bel.sort_values(by='dist to opt', inplace=True)
    df_bel['dist to opt order'] = np.arange(len(df_bel))
    df_bel.loc[df_bel[ycol]<yval+yrange]
    xrange = xval
    for i, s in df_bel.iterrows():
        if s[ycol] < yval+yrange:
            xrange = s[xcol]
        else:
            break
    return np.abs(xrange - xval)


def compute_emp_min(L_tradeoff, err_fit, err_range=0.01, groupby='pc'):
    """
    Computes the empirical pc the minimizes the reconstruction error
    """
    L_by_B = L_tradeoff.groupby(L_tradeoff['B'].apply(lambda x: round(x, 6)))#.mean()
    
    emp_min = []
    for B, sL_by_B in L_by_B:
        sL_by_B_pc = sL_by_B.groupby(groupby)[err_fit].mean().reset_index()
        idxmin = sL_by_B_pc[err_fit].idxmin()
        ssL_by_B_pc = sL_by_B_pc.loc[idxmin]
        pc = ssL_by_B_pc[groupby]
        is_max = pc == sL_by_B_pc[groupby].max()
        err = ssL_by_B_pc[err_fit]
        pc_range_bel = continuously_in_range(sL_by_B_pc, xcol=groupby, ycol=err_fit, xval=pc, yval=err, yrange=err_range, below=True)
        pc_range_abv = continuously_in_range(sL_by_B_pc, xcol=groupby, ycol=err_fit, xval=pc, yval=err, yrange=err_range, below=False)
        emp_min.append({'B':B, groupby:pc, err_fit:err, 
                        groupby + '_range_bel': pc_range_bel, 
                        groupby + '_range_abv': pc_range_abv, 
                        'is_max': is_max})
        
    emp_min = pd.DataFrame(emp_min)

    return emp_min


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
        L_per_B.append(msL.rolling(roll).mean().iloc[roll-1:].reset_index())
    rL = pd.concat(L_per_B)
    return rL