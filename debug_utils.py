from foghaze_generator.midas_dmap_estimator import MidasDmapEstimator
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os


"""
Debugging Notes:

1/ OpenCV read() an image file path into a numpy array which is 8-bit uint and BGR
2/ When working with OpenCV functions (methods), need to convert image into BGR
3/ Matplotlib read() an image file path into a numpy array which is 8-bit uint and RGB - Matplotlib utilizes Pillow in lower layer
4/ Pillow works with RGB
"""


"""
@param (np.ndarray) img
@param (bool) print_img
"""
def print_img_info(img, print_img=False):
    print(
        'Image information:',
        f'(height, width, channel): {img.shape}',
        f'Dtype: {img.dtype}',
        f'Min value: {np.amin(img)}',
        f'Max value: {np.amax(img)}',
        sep='\n')
    
    if print_img:
        print(img)
    print()


"""
@param (np.ndarray) img
@param (bool) write_to_file
@return np.ndarray
"""
def estimate_depth_map(img, write_to_file=False):
    dmap_estimator = MidasDmapEstimator([img])
    dmap = dmap_estimator.estimate_depth_maps()[0]
    # normalized_dmap = cv.normalize(dmap, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    # bgr_dmap = cv.cvtColor(normalized_dmap, cv.COLOR_GRAY2BGR)

    if write_to_file:
        if os.path.exists('debug_dmap.jpg'):
            os.remove('debug_dmap.jpg')
        plt.imsave('debug_dmap.jpg', dmap)

    return dmap


dmap = estimate_depth_map('test_img.jpg', True)
print_img_info(dmap)
