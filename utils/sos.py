import numpy as np


def sos(ksfull):
    row, col, coils = ksfull.shape
    return np.sqrt(np.sum(np.abs(ksfull)**2, axis=2))