import scipy.io as sio
import cv2


def load_data(ksfull_path, mask_path):
    '''
    This function is for loading the ksfull data and mask
    :param ksfull_path:
    :param mask_path:
    :return:
    '''
    ksfull = sio.loadmat(ksfull_path)
    mask = cv2.imread(mask_path)
    return ksfull, mask[:, :, 1]
