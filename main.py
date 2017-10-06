import sys
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
sys.path.append('data/mnist')
import mnist_basics as mb

# 60k different images,
def load_dataset():
    images, labels = mb.load_mnist()
    return images, labels

def get_batches(images, labels, batch_size):
    n_datapoints = images.shape[0]
    ### X = []
    ### for i, each in enumerate(images):
    ###     # each = each / each.sum(axis=1)
    ###     each_norm = np.apply_along_axis(lambda x: x / x.sum(), 1, each)
    ###     each_norm = np.nan_to_num(each_norm)
    ###     X.append(each_norm)
    ###     print(i)

    ### print('done with this shit')
    ### images = np.array(X)
    images_flattened = images.reshape([n_datapoints, -1])
    prev = 0
    for _next in range(batch_size, n_datapoints, batch_size):
        yield images_flattened[prev:_next], labels[prev:_next].reshape([-1])
        prev = _next

# X = get_batches(images, labels, 5)


# M = [1, 2
#      3, 4]

# M.sum(axis=1)
