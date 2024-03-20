import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_mean_bucket_exp(X, pseudo, n_buckets=5, plot=False):
    """
    Compute the expression profile based on bucket assignment
    X - expression
    bucket - bucket assignment
    """
    # check pseudo is valid
    if isinstance(pseudo, pd.Series):
        n_na = pseudo.isna().sum()
        n_null = pseudo.isnull().sum()
        n_inf = np.isinf(pseudo).sum()
        n_unique = pseudo.nunique()
        if n_na > 0 or n_null > 0 or n_inf > 0:
            print(f'Pseudo contains {n_na} NA, {n_null} NULL, {n_inf} INF. There are {n_unique} unique values. ')
            raise ValueError('Pseudo contains invalid values')
    bucket = pd.qcut(pseudo, q=n_buckets, labels=np.arange(n_buckets), duplicates='drop')
    lX = np.log1p(X)
    lX_bucket = pd.concat((lX, bucket), axis=1)
    
    bucket_mean = lX_bucket.groupby(bucket.name).mean()
    norm_bucket_mean = bucket_mean / bucket_mean.max(axis=0)
    
    norm_bucket_mean.fillna(0, inplace=True)
    if plot:
        ngenes = min(10, X.shape[1])
        for i in range(ngenes):
            plt.plot(bucket_mean.index, bucket_mean[bucket_mean.columns[i]])
        plt.show()
    return norm_bucket_mean


def expression_correlation(X, sX):
    """
    Compute the correlation between the expression and the reconstructed expression
    X - expression
    sX - reconstructed expression
    :return: 
    correlation
    """
    corr_list = []
    for g in X.columns:
        corr_list.append(np.corrcoef(X[g], sX[g])[0,1])
    print(f'Number of nans: {np.isnan(corr_list).sum()}')
    return np.nanmean(corr_list)
