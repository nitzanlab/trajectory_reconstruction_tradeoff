import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_mean_bucket_exp(X, pseudo, n_buckets=5, plot=False):
    """
    Compute the expression profile based on bucket assignment
    X - expression
    bucket - bucket assignment
    """
    bucket = pd.qcut(pseudo, q=n_buckets, labels=np.arange(n_buckets), duplicates='drop')
    lX = np.log1p(X)
    tmp = pd.concat((lX, bucket), axis=1)
    
    bucket_mean = tmp.groupby(bucket.name).mean()
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
    return np.nanmean(corr_list)
