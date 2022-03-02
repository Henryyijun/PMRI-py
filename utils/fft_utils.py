import numpy as np
from numpy.fft import fft2
from numpy.fft import ifft2


def ifft_2d(ksfull):
    _, _, coils = ksfull.shape
    res = np.zeros(ksfull.shape, dtype='complex128')
    for i in range(coils):
        res[:, :, i] = ifft2(ksfull[:, :, i])
    return res


def fft_2d(ksfull):
    _, _, coils = ksfull.shape
    res = np.zeros(ksfull.shape, dtype='complex128')
    for i in range(coils):
        res[:, :, i] = fft2(ksfull[:, :, i])
    return res




