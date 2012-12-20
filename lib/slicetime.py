
import numpy as np

colon_band = np.array([
    [  0,   0,   0],
    [  1,   1,   1],
    [  0,   0,   0],
    [  0,   0,   0],
    [  0,   0,   0],
    [136, 224, 167],
    [ 47,  77,  57],
    [  0,   0,   0],
    [  0,   0,   0],
    [ 47,  77,  57],
    [136, 224, 167],
    [  0,   0,   0],
    [  0,   0,   0]], dtype=np.uint8)

colon_positions = 29, 40
colon_mad_max = 1000

class ColonMissingError(ValueError):
    pass

def find_colon_position(img):
    delta = np.abs(img - colon_band[::, np.newaxis, ::]).sum(axis=2).sum(axis=0)
    l = np.argmin(delta)
    m = delta[l]
    if m > colon_mad_max or l not in colon_positions:
        raise ColonMissingError
    return l

def slice_time(img):
    l = find_colon_position(img)
    acc_section = []
    for off in [-1,1]:
        acc_digits = []
        for i in [0,1]:
            a = l + off * (i*11 + 3)
            b = a + off*8
            acc_digits.append(img[::, min(a,b):max(a,b)+1, ::])
        if off == -1:
            acc_digits = acc_digits[::-1]
        acc_section.append(acc_digits)
    return acc_section


