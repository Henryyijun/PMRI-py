from get_calibration import *

if __name__ == '__main__':
    import cv2
    mask = cv2.imread('E:\\DeepLearning\\PMRI\\mask\\mask\\512x512\\CartesianMask512_0_15_1.png')
    mask = mask[:, :, 0]
    acs_size, acs_range = get_calibration(np.rot90(mask))
    print(acs_size)
    print(acs_range)
