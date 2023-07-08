import cv2 as cv
import math
import numpy as np
import utilities.utilities as utils


DEFAULT_DTYPE = np.float32

DEFAULT_PATCH_SIZE = 15
DEFAULT_OMEGA = 0.95
DEFAULT_T0 = 0.1
DEFAULT_RADIUS = 60
DEFAULT_EPS = 0.0001
DEFAULT_FUSION_WEIGHT = 0.5


# Dark channel is a statistics calculated from a local patch scanning a color image
def _dark_channel(image_3c: np.ndarray, patch_size: int) -> np.ndarray:
    c1, c2, c3 = cv.split(image_3c)
    min_3c = cv.min(cv.min(c1, c2), c3)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv.erode(min_3c, kernel)

    return dark_channel


def _atm_light(image_3c: np.ndarray, dark_channel: np.ndarray) -> np.ndarray:
    height, width, _ = image_3c.shape
    total_pixels = height * width

    # Pick the top 0.1% brightest pixels in the dark channel, those contributes to the most haze-opaque area
    num_candidates = max(math.floor(total_pixels * 0.001), 1)
    dc_vector = dark_channel.flatten()
    candidate_indices = dc_vector.argsort()[total_pixels - num_candidates::]

    atm_light = 0
    image_3c_vectors = image_3c.reshape(total_pixels, 3) # flatten each channel into vector
    for i in range(num_candidates):
        atm_light = np.maximum(atm_light, image_3c_vectors[candidate_indices[i]])

    return atm_light


# Refine transmission map using guided image filter
def _refined_tmap(bgr_image: np.ndarray, base_tmap: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    # Ensure dtype "float32" to be compatible with cv.ximgproc.guidedFilter()
    gray_guidance = cv.cvtColor(bgr_image.astype(np.float32), cv.COLOR_BGR2GRAY)
    refined_tmap = cv.ximgproc.guidedFilter(gray_guidance, base_tmap.astype(np.float32), radius, epsilon)

    return refined_tmap.astype(DEFAULT_DTYPE)


def defoghaze(
    bgr_image: np.ndarray,                  # expect image to be BGR, uint8, values are within [0, 255]
    patch_size: int = DEFAULT_PATCH_SIZE,   # size of patch aka window to construct dark channel
    omega: float = DEFAULT_OMEGA,           # control small amount of haze at distant objects
    t0: float = DEFAULT_T0,                 # control lower bound of transmission map
    radius = DEFAULT_RADIUS,                # radius of guided filter
    epsilon = DEFAULT_EPS,                  # regularization term of guided filter
    fusion_weight = DEFAULT_FUSION_WEIGHT
) -> dict:
    
    normalized_bgr = utils.minmax_normalize(bgr_image, new_dtype=DEFAULT_DTYPE)

    # Build pyramid of BGR images, downsample the original until the size is less than the patch size
    bgr_pyramid = []
    current_level = normalized_bgr
    while min(current_level.shape[:2]) >= patch_size:
        bgr_pyramid.append(current_level)
        current_level = cv.pyrDown(current_level)

    dark_channels = [_dark_channel(img, patch_size) for img in bgr_pyramid]
    atm_light = np.max(
        [_atm_light(img, dark_channels[i]) for i, img in enumerate(bgr_pyramid)],
        axis=0
    )

    base_tmaps = []
    for i, img in enumerate(bgr_pyramid):
        tmap = 1 - omega * _dark_channel(img / atm_light, patch_size)
        base_tmaps.append(tmap)
    
    # fusion of transmission maps
    base_tmap = None
    refined_tmap = None
    fused_tmap = base_tmaps[-1]
    for i, img in enumerate(reversed(bgr_pyramid)):
        if i != 0:
            base_tmap = base_tmaps[-i-1]
            refined_tmap = cv.pyrUp(refined_tmap)

            if refined_tmap.shape != base_tmap.shape:
                refined_tmap = cv.resize(refined_tmap, base_tmap.shape[::-1])

            fused_tmap = cv.addWeighted(base_tmap, 1-fusion_weight, refined_tmap, fusion_weight, 0.0)

        refined_tmap = _refined_tmap(bgr_pyramid[-i-1], fused_tmap, radius, epsilon)
    
    recovered_bgr = (normalized_bgr - atm_light) / np.maximum(refined_tmap.reshape(*refined_tmap.shape, 1), t0) + atm_light
    recovered_bgr = np.clip(recovered_bgr, 0, 1)

    return {
        'dark_channel': dark_channels[0],
        'atm_light': atm_light,
        'base_tmap': base_tmaps[0],
        'refined_tmap': refined_tmap,
        'recovered_bgr': recovered_bgr
    }
