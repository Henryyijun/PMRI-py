import numpy as np

from mask_utils.get_calibration import get_calibration
from utils.fft_utils import *
from utils.sos import sos
from utils.wthresh import soft_thresh
import matplotlib.pyplot as plt


class Reconstruction:
    def __init__(self, data, mask, algorithm):
        self.algorithm = algorithm
        self.data = data
        self.mask = mask
        self.acs_size, self.acs_location = get_calibration(self.mask)
        self.__sensitivity = self.__get_sensitivity()

    def __get_sensitivity(self):
        """
        This function is used for estimating the initial sensitivity map.
        :return:
        """
        row, col, coils = self.data.shape
        sensitivity_mask = np.zeros((row, col, coils))
        sensitivity_mask[self.acs_location[0]:self.acs_location[1], :, :] = 1
        data_center = self.data * sensitivity_mask
        sensitivity = self.__sensitivity_compute(data_center)
        return sensitivity

    def __sensitivity_compute(self, data_center):
        image_zero_fill = ifft_2d(data_center)
        image_sos = sos(image_zero_fill)
        sensitivity = image_zero_fill / np.dstack([image_sos] * data_center.shape[-1])
        return sensitivity

    def get_sensitivity(self):
        return self.__sensitivity

    def reconstruct(self, max_iter, regularization=None, mu=0):
        """
        :param max_iter: The max iteration number of the algorithm
        :param regularization: The regularization method, l1, l2, TV, wavelet, tight frame
        :param mu: The positive regularization parameter
        :return: The reconstruction image
        """
        if regularization is not None:
            if regularization == 'l2':
                if self.algorithm.lower() == 'cg':
                    return self.__l2_cg_reconstruct(max_iter, mu)
            elif regularization == 'l1':
                if self.algorithm == 'fista':
                    return self.__fista_reconstruct(max_iter, mu)
            elif regularization == 'wavelet':
                pass
            elif regularization == 'framelet':
                pass

        else:
            if self.algorithm.lower() == 'cg':
                return self.__cg_reconstruct(max_iter)
            else:
                pass

    def __cg_reconstruct(self, max_iter):
        """
        This function uses conjugate gradient methods to solve the pMRI problem
        PFSx = y => Ax = y, the normal equation is AtAx = Aty => Bx = b.
        :param max_iter: The iteration number of the algorithm
        :return: reconstruction image.
        """
        mask = np.dstack([self.mask] * self.data.shape[-1])
        A = Matrix(mask, self.__sensitivity)
        y = mask * self.data
        x = np.zeros(ifft_2d(y).shape[:2])
        B = lambda xx: A.mul_t(A.mul(xx))
        b = A.mul_t(y)

        rk = b - B(x)
        pk = rk

        for i in range(max_iter):
            print("The %d iteration" % i)
            Bpk = B(pk)
            ak = np.dot(np.conj(rk.flatten()), rk.flatten()) / np.dot(np.conj(pk).flatten(), Bpk.flatten())
            xk = x + ak * pk
            rk_1 = rk - ak * B(pk)
            if np.linalg.norm(rk_1.flatten()) < 1E-2:
                break

            bk = np.dot(np.conj(rk_1.flatten()), rk_1.flatten()) / np.dot(np.conj(rk.flatten()), rk.flatten())
            pk = rk_1 + bk * pk
            rk = rk_1
            x = np.abs(xk)

        return x

    def __l2_cg_reconstruct(self, max_iter, mu):
        """
        This function uses conjugate gradient methods with l2 regularization to solve the pMRI problem
        PFSx = y => Ax = y, the normal equation is AtAx + mu x= Aty => Bx = b, where B = (AtA+mu)
        :param max_iter: The iteration number of the algorithm
        :return: reconstruction image.
        """
        mask = np.dstack([self.mask] * self.data.shape[-1])
        A = Matrix(mask, self.__sensitivity)
        y = mask * self.data
        x = np.zeros(ifft_2d(y).shape[:2])
        B = lambda xx: A.mul_t(A.mul(xx)) + mu * xx
        b = A.mul_t(y)
        rk = b - B(x)
        pk = rk

        for i in range(max_iter):
            print("The %d iteration" % i)
            Bpk = B(pk)
            ak = np.dot(np.conj(rk.flatten()), rk.flatten()) / np.dot(np.conj(pk).flatten(), Bpk.flatten())
            xk = x + ak * pk
            rk_1 = rk - ak * B(pk)
            if np.linalg.norm(rk_1.flatten()) < 1E-2:
                break

            bk = np.dot(np.conj(rk_1.flatten()), rk_1.flatten()) / np.dot(np.conj(rk.flatten()), rk.flatten())
            pk = rk_1 + bk * pk
            rk = rk_1
            x = np.abs(xk)

        return x

    def __fista_reconstruct(self, max_iter, mu):
        """
        Fast Iterative Soft Thresholding Algorithm(FISTA),
        problem: ||Ax-b||^2 + mu*||x||_1, gradient: At(Ax-b)
        :param max_iter:
        :param mu:
        :return: The reconstruction image.
        """
        mask = np.dstack([self.mask] * self.data.shape[-1])
        A = Matrix(mask, self.__sensitivity)
        b = mask * self.data
        l = 1
        tk = 1
        xk = sos(ifft_2d(b))
        yk = xk.copy()

        for i in range(max_iter):
            print("The %d iteration" % i)
            # print(self.gradient(A, yk, b))
            xk1 = soft_thresh(yk - self.gradient(A, yk, b)/l, mu/l)
            tk1 = 1/2 + np.sqrt(1 + 4*tk**2)/2
            yk1 = xk1 + (tk-1)/tk1 * (xk1 - xk)
            # update
            xk = np.abs(xk1.copy())
            tk = tk1.copy()
            yk = np.abs(yk1.copy())
        return np.abs(xk)

    def gradient(self, A, xx, b):
        return A.mul_t(A.mul(xx) - b)


class Matrix:
    """
    This class implement the system matrix A, where A = PFS
    Author:Henry
    Date:2022/3/2
    """
    def __init__(self, mask, sensitivity):
        self.mask = mask
        self.sensitivity = sensitivity

    def mul(self, x):
        """
        This function implements the A*x,  where A=PFS
        :param x: The MRI image, 2d array
        :return: The under-sampled k-space data
        """
        row, col, coils = self.sensitivity.shape
        xx = np.dstack([x] * coils)
        y = self.mask * fft_2d(self.sensitivity * xx)
        return y

    def mul_t(self, x):
        """
        This function implements the At*x,  where A=PFS, StFtPt x
        :param x: The under-sampled k-space data
        :return: Atx
        """
        yy = ifft_2d(self.mask * x)
        y = np.conj(self.sensitivity) * yy
        return np.sum(y, axis=2)
