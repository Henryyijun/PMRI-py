import matplotlib.pyplot as plt
import numpy as np

from load_data import load_data
from utils.sos import sos
from utils.fft_utils import fft_2d, ifft_2d
from utils.imshow import imshow

if __name__ == '__main__':
    ksfull_path = 'E:\\DeepLearning\\PMRI\\data\\512x512\\05_t2_tse_tra_512_s33_3mm_13.mat'
    mask_path =  'E:\\DeepLearning\\PMRI\\mask\\mask\\512x512\\CartesianMask512_0_15_1.png'

    ksfull, mask = load_data(ksfull_path, mask_path)
    ksfull = ksfull['ksfull']

    mask = np.dstack([mask]*ksfull.shape[-1])
    mask = np.rot90(mask)
    un_ksfull = mask * ksfull
    image = sos(ifft_2d(un_ksfull))
    imshow(image, mask[:,:,0])
