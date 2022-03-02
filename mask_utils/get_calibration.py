import numpy as np


def get_calibration(mask):
    '''
    This function use mask to get the calibration location.
    :param mask: 2D mask
    :return: calibration_size, ACS_location
    '''
    rows, cols = mask.shape
    row_center = rows // 2
    col_center = cols // 2
    sx = sy = 1
    row_pos = row_center - 1
    while row_pos >= 0:
        if mask[row_pos, col_center] == 0:
            break
        else:
            sx += 1
        row_pos -= 1

    acs_up_pos = row_pos + 1
    row_pos = row_center + 1
    while row_pos < rows:
        if mask[row_pos, col_center] == 0:
            break
        else:
            sx += 1
        row_pos += 1

    acs_down_edge = row_pos - 1
    # col
    col_pos = col_center - 1
    while col_pos >= 0:
        if mask[row_center, col_pos] == 0:
            break
        else:
            sy += 1
        col_pos -= 1

    col_pos = col_center + 1
    while col_pos < cols:
        if mask[row_center, col_pos] == 0:
            break
        else:
            sy += 1
        col_pos += 1

    return np.array([sx, sy]), np.array([acs_up_pos, acs_down_edge])
