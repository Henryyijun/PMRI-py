import numpy as np
from utils.load_data import load_data
import matplotlib.pyplot as plt

from reconstruction.reconstruction import Reconstruction

if __name__ == '__main__':
    ksfull_path = 'E:\\DeepLearning\\PMRI\\data\\512x512\\05_t2_tse_tra_512_s33_3mm_13.mat'
    mask_path =  'E:\\DeepLearning\\PMRI\\mask\\mask\\512x512\\CartesianMask512_0_15_1.png'
    ksfull, mask = load_data(ksfull_path, mask_path)
    ksfull = ksfull['ksfull']
    mask = np.rot90(mask)

    rec = Reconstruction(None, ksfull, mask)

    plt.imshow(abs(rec.get_sensitivity())[:, :, 0], cmap='gray')
    plt.show()
