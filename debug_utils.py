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


def normalize_depth_map(dmap, inverse=False):
    normalized_dmap = cv.normalize(dmap, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    if inverse:
        normalized_dmap = 255 - normalized_dmap
    
    return normalized_dmap.astype(np.uint8)


"""
@param (np.ndarray) img
@param (bool) write_to_file
@return np.ndarray
"""
def estimate_depth_map(img, write_to_file=False):
    dmap_estimator = MidasDmapEstimator([img])
    dmap = dmap_estimator.estimate_depth_maps()[0]
    normalized_dmap = normalize_depth_map(dmap, False)

    if write_to_file:
        if os.path.exists('debug_dmap_grey.jpg'):
            os.remove('debug_dmap_grey.jpg')
        if os.path.exists('debug_dmap_magma.jpg'):
            os.remove('debug_dmap_magma.jpg')
        if os.path.exists('debug_dmap_viridis.jpg'):
            os.remove('debug_dmap_viridis.jpg')

        cv.imwrite('debug_dmap_grey.jpg', cv.cvtColor(normalized_dmap, cv.COLOR_GRAY2BGR))
        plt.imsave('debug_dmap_viridis.jpg', normalized_dmap, cmap='viridis')
        plt.imsave('debug_dmap_magma.jpg', normalized_dmap, cmap='magma')

    return normalized_dmap


if __name__ == "__main__":
    dmap = estimate_depth_map('test_img.jpg', True)
    print_img_info(dmap, True)
    plt.imshow(dmap)
    plt.show()
