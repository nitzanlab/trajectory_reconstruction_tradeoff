import numpy as np

def find_min(x, y):
    """
    Find pc that minimizes the error
    :param x:
    :param y:
    :return:
    """
    return x[y == np.min(y)]
