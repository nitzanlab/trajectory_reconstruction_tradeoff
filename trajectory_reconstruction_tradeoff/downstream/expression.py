import numpy as np
import pandas as pd

# from scipy.interpolate import UnivariateSpline
#
# def interp(X, knots=5):
#     """
#     Fits and predicts expression as UnivariateSpline
#     :param X:
#     :param knots:
#     :return:
#     """
#     X_interp = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
#     for gcol in X.columns:
#         idx = X[~X[gcol].isna()].index
#         if len(idx) == 0:
#             continue
#         x = X.loc[idx].index
#         y = X.loc[idx, gcol]
#         f = UnivariateSpline(x, y, k=knots) #fill_value="extrapolate")kind=5, bounds_error=False
#         X_interp[gcol] = f(X.index)
#     return X_interp


# Mean expression given groupings
def get_mean_bucket_exp(X, pseudo, n_buckets=10):
    """
    Compute the expression profile based on bucket assignment
    X - expression
    bucket - bucket assignment
    """
    bucket = pd.qcut(pseudo, q=n_buckets, labels=np.arange(n_buckets))
    lX = np.log1p(X)
    tmp = pd.concat((lX, bucket), axis=1)
    # log1p?
    bucket_mean = tmp.groupby(bucket.name).mean()
    norm_bucket_mean = bucket_mean / bucket_mean.max(0)
    # TODO: how to handle???
    norm_bucket_mean.fillna(0, inplace=True)
    return norm_bucket_mean


def expression_correlation(X, sX):
    """
    """
    corr_list = []
    for g in X.columns:
        corr_list.append(np.corrcoef(X[g], sX[g])[0,1])
    return np.nanmean(corr_list)
