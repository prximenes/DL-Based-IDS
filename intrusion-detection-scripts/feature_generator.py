import gc
from sys import getsizeof
import numpy as np

def split_into_nibbles(array):
    array = np.array(array)
    shape = list(array.shape)
    shape[-1] *= 2
    array2 = (array.reshape(-1, 1) & np.array([0xF0, 0x0F], dtype=np.uint8)) >> np.array([4, 0], dtype=np.uint8)
    return array2.reshape(shape)

def create_dataset(dataset, ypsilons, look_back=44, size_bytes=58):
    X, Y = [], []
    look_back += 1

    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        X.append(a)
        if np.all(np.array(ypsilons[i:(i+look_back)]) == 0):
            Y.append(0)
        else:
            Y.append(1)

    features = []
    for i in range(len(X)):
        deltaX = []
        for e in range(look_back-1):  
            e += 1
            b = split_into_nibbles(
                [((a_i - b_i) % 256) for a_i, b_i in list(zip(X[i][e][:size_bytes], X[i][e-1][:size_bytes]))]
            )       
            deltaX.append(b)
        features.append(np.array(deltaX, dtype=np.uint8))
    return np.array(features), np.array(Y)
