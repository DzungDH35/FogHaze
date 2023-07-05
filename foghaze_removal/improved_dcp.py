"""
1/ Algorithm will work with normalized image, which has value within [0, 1] and dtype = float64

"""

import numpy as np
import cv2 as cv
import math


DEFAULT_PATCH_SIZE = 15
DEFAULT_OMEGA = 0.95
DEFAULT_T0 = 0.1
DEFAULT_RADIUS = 60
DEFAULT_EPS = 0.0001


def scale_array(arr: np.ndarray, old_range: tuple = None, new_range: tuple = (0, 1)) -> np.ndarray:
    if old_range:
        low_old, high_old = old_range
    else:
        low_old = np.min(arr)
        high_old = np.max(arr)

    low_new, high_new = new_range

    return low_new + (arr - low_old) * (high_new - low_new) / (high_old - low_old)


# Normalize image to [0, 1] and use dtype 'float32'
def _normalized_image(image: np.ndarray) -> np.ndarray:
    dtype = image.dtype
    image = image.astype(np.float32)

    if dtype == np.uint8:
        return image / 255
    elif dtype == np.uint16:
        return image / 65535
    elif dtype not in [np.float32, np.float64]:
        raise TypeError('Not supported image type!')
    
    return image
        

def _dark_channel(image_3c: np.ndarray, patch_size: int) -> np.ndarray:
    c1, c2, c3 = cv.split(image_3c)
    min_3c = cv.min(cv.min(c1, c2), c3)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv.erode(min_3c, kernel)

    return dark_channel


def _atm_light(image_3c: np.ndarray, dark_channel: np.ndarray) -> np.ndarray:
    height, width, channels = image_3c.shape
    total_pixels = height * width

    # Pick the top 0.1% brightest pixels in the dark channel
    num_candidates = max(math.floor(total_pixels * 0.001), 1)
    dc_vector = dark_channel.flatten()
    candidate_indices = dc_vector.argsort()[total_pixels - num_candidates::]

    atm_light = 0
    image_3c_vectors = image_3c.reshape(total_pixels, 3)
    for i in range(num_candidates):
        atm_light += image_3c_vectors[candidate_indices[i]]

    return atm_light/num_candidates


# Refine transmission map using guided image filter
def _refined_tmap(bgr_image: np.ndarray, base_tmap: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    gray_guidance = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)
    refined_tmap = cv.ximgproc.guidedFilter(gray_guidance, base_tmap, radius, epsilon)

    return refined_tmap


def defoghaze(
    bgr_image: np.ndarray,
    patch_size: int = DEFAULT_PATCH_SIZE,
    omega: float = DEFAULT_OMEGA,           # control small amount of haze at distant objects
    t0: float = DEFAULT_T0,                 # control lower bound of transmission map
    radius = DEFAULT_RADIUS,                # radius of guided filter
    epsilon = DEFAULT_EPS                   # regularization term of guided filter
) -> dict:
    
    normalized_bgr = _normalized_image(bgr_image)
    bgr_pyramid = []

    current_level = normalized_bgr
    while min(current_level.shape[:2]) > patch_size:
        bgr_pyramid.append(current_level)
        current_level = cv.pyrDown(current_level)
    
    # debug
    print(len(bgr_pyramid))
    [print(i.shape) for i in bgr_pyramid]
    print()

    dark_channels = [_dark_channel(img, patch_size) for img in bgr_pyramid]
    atm_lights = [_atm_light(img, dark_channels[i]) for i, img in enumerate(bgr_pyramid)]

    base_tmaps = []
    for i, img in enumerate(bgr_pyramid):
        tmap = 1 - omega * _dark_channel(img / atm_lights[i], patch_size)
        base_tmaps.append(tmap)

    # fusion algorithm
    refined_tmap = None
    for i, img in enumerate(reversed(bgr_pyramid)):
        fused_tmap = base_tmaps[-i-1]

        if i != 0:
            upsampled = cv.pyrUp(refined_tmap)

            if upsampled.shape[0] != fused_tmap.shape[0] or upsampled.shape[1] != fused_tmap.shape[1]:
                upsampled = cv.resize(upsampled, (fused_tmap.shape[1], fused_tmap.shape[0]))
            
            alpha = 0.3
            fused_tmap += cv.addWeighted(fused_tmap, 1-alpha, upsampled, alpha, 0.0)

        refined_tmap = _refined_tmap(bgr_pyramid[-i-1], fused_tmap, radius, epsilon)
    
    atm_light = np.max(atm_lights, axis=0)
    refined_tmap = scale_array(refined_tmap)
    recovered_bgr = (normalized_bgr - atm_light) / np.maximum(refined_tmap.reshape(*refined_tmap.shape, 1), t0) + atm_light
    recovered_bgr = np.clip(recovered_bgr, 0, 1)

    return {
        'dark_channel': dark_channels[0],
        'atm_light': atm_light,
        'base_tmap': base_tmaps[0],
        'refined_tmap': refined_tmap,
        'recovered_bgr': recovered_bgr
    }
