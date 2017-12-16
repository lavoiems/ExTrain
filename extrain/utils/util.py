import numpy as np


def transpose(m):
    """
    Transpose a matrix
    :param m: matrix (a list of list)
    :return: Transposed list
    """
    try:
        return map(list, zip(*m))
    except TypeError:
        raise TypeError('Argument passed is not a matrix (a list of list)')


def shuffle(data):
    idx = np.random.permutation(len(data))
    return data[idx]
