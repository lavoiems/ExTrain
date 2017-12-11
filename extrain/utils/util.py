import numpy as np

def transpose(m):
    """
    Transpose a matrix
    :param l: matrix
    :return: Transposed list
    """
    return map(list, zip(*m))


def shuffle(data):
    idx = np.random.permutation(len(data))
    return data[idx]
