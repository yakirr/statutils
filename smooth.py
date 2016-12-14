from __future__ import print_function, division
import numpy as np

def smooth(vec, window_size, stride=None):
    if stride is None:
        stride = window_size / 20
    result = np.zeros(vec.shape)
    result = np.array([
        vec[i:i+window_size].mean()
        for i in np.arange(1,len(vec)-window_size, stride)])
    return result
