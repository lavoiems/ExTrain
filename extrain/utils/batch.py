import numpy as np
from extrain.utils.util import transpose, shuffle


def _batcher(n, iterable):
    """
    Create minibatches
    :param n: Number of element per minibatches
    :param iterable:
    :return: List of minibatches
    """
    args = [iter(iterable)] * n
    return [transpose(batch) for batch in zip(*args)]


def batch(batch_size, datas):
    """
    Prepare mini batch with no labels
    """
    datas = transpose([shuffle(data) for data in datas])
    return _batcher(batch_size, datas)


def batch_label(batch_size, datas):
    """
    Prepare mini batch with labels
    """
    datas, labels = datas[0], datas[1]
    assert len(datas) == len(labels)

    all_data = []
    all_labels = []

    for data, label in zip(datas, labels):
        idx = np.random.permutation(len(data))
        all_data.append(data[idx])
        all_labels.append(label[idx])

    return _batcher(batch_size, transpose(all_data + all_labels))
