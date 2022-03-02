from mask_utils.get_calibration import get_calibration
from utils.fft_utils import *
from utils.sos import sos

class Reconstruction:
    def __init__(self, algorithm, data, mask):
        self.algorithm = algorithm
        self.data = data
        self.mask = mask
        self.acs_size, self.acs_location = get_calibration(self.mask)
        self.__sensitivity = self.__get_sensitivity()

    def __get_sensitivity(self):
        '''
        This function is used for estimating the initial sensitivity map.
        :return:
        '''
        row, col, coils = self.data.shape
        sensitivity_mask = np.zeros((row, col, coils))
        sensitivity_mask[self.acs_location[0]:self.acs_location[1], :, :] = 1
        data_center = self.data * sensitivity_mask
        sensitivity = self.__sensitivity_compute(data_center)
        return sensitivity

    def __sensitivity_compute(self, data_center):
        image_zero_fill = ifft_2d(data_center)
        image_sos = sos(image_zero_fill)
        sensitivity = image_zero_fill / np.dstack([image_sos]*data_center.shape[-1])
        return sensitivity

    def get_sensitivity(self):
        return self.__sensitivity

    def reconstruct(self, max_iter):
        if self.algorithm.lower() == 'cg':
            return self.__cg_reconstruct(max_iter)
        else:
            pass

    def __cg_reconstruct(self, max_iter):
        '''
        This function uses  conjugate gradient methods to solve the pMRI problem
        :param max_iter:
        :return:
        '''
        mask = np.dstack([self.mask]*self.data.shape[-1])
