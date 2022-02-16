import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score


def cluster(X, D, expr_red, meta=None, by='kmeans', n_clusters=None):
    # if (meta is not None) and (by in meta.columns):
    #     return meta[by]
    if by == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(expr_red)
        cls = kmeans.labels_.astype(str)
    else:
        print('Clustering %s not in metadata or not implemented.' % by)
    return pd.DataFrame({by: cls}, index=X.index)



def compute_f1(cluster_labels1, cluster_labels2):
    """

    :param cluster_labels1:
    :param cluster_labels2:
    :return:
    """
    jaccard = pd.DataFrame(0, columns=cluster_labels1.unique(), index=cluster_labels2.unique())
    for col in jaccard.columns:
        for row in jaccard.index:
            jaccard.loc[row, col] = jaccard_score(y_true=cluster_labels1 == col, y_pred=cluster_labels2 == row)

    recovery = jaccard.max().mean()
    relevance = jaccard.max(1).mean()
    f1 = 2 / (1 / recovery + 1 / relevance)
    return f1
