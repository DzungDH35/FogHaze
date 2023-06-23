import numpy as np
import cv2 as cv
import math


DEFAULT_PATCH_SIZE = 15
DEFAULT_OMEGA = 0.95
DEFAULT_T0 = 0.1
DEFAULT_RADIUS = 60
DEFAULT_EPS = 0.0001


# Normalize image to [0, 1] and use dtype 'float32'
def _normalized_image(image):
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


def defoghaze(
    bgr_image: np.ndarray,
    patch_size: int = DEFAULT_PATCH_SIZE,
    omega: float = DEFAULT_OMEGA,   # control small amount of haze at distant objects
    t0: float = DEFAULT_T0,         # control lower bound of transmission map
    radius = DEFAULT_RADIUS,        # radius of guided filter
    epsilon = DEFAULT_EPS           # regularization term of guided filter
) -> dict:
    
    normalized_bgr = _normalized_image(bgr_image)
    dark_channel = _dark_channel(normalized_bgr, patch_size)
    atm_light = _atm_light(normalized_bgr, dark_channel)

    # Estimate and refine transmission map
    base_tmap = 1 - omega * _dark_channel(normalized_bgr / atm_light, patch_size)
    gray_image = _normalized_image(cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY))
    refined_tmap = cv.ximgproc.guidedFilter(gray_image, base_tmap, radius, epsilon)

    recovered_bgr = (normalized_bgr - atm_light) / np.maximum(refined_tmap.reshape(*refined_tmap.shape, 1), t0) + atm_light
    np.clip(recovered_bgr, 0, 1, recovered_bgr)

    return {
        'dark_channel': dark_channel,
        'atm_light': atm_light,
        'base_tmap': base_tmap,
        'refined_tmap': refined_tmap,
        'recovered_bgr': recovered_bgr
    }
