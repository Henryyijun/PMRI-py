import numpy as np


def soft_thresh(x, mu):
    return np.sign(x) * np.maximum(np.abs(x) - mu, 0)


# def soft_thresh(x, mu):
#     real = np.real(x)
#     imag = np.imag(x)
#     soft_real = __soft_thresh(real, mu)
#     soft_imag = __soft_thresh(imag, mu)
#     return soft_real + 1j * soft_imag



