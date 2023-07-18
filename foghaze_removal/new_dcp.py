import concurrent.futures
import cv2 as cv
import math
import numpy as np
import utilities.utilities as utils
import multiprocessing
import os
from time import perf_counter


DEFAULT_DTYPE = np.float32

DEFAULT_PATCH_SIZE = 15
DEFAULT_OMEGA = 0.95
DEFAULT_T0 = 0.1
DEFAULT_RADIUS = 60
DEFAULT_EPS = 0.0001

        
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


def _improved_atm_light(image_3c: np.ndarray, omega_size: int):
    process = multiprocessing.current_process()
    print(f'Starting algorithm in process {process}, pid = {os.getpid()}, omega_size = {omega_size}')
    height, width, _  = image_3c.shape
    psi_size = omega_size * 4
    half_psi_size = psi_size // 2
    is_psi_size_even = psi_size % 2 == 0

    atm_light = np.zeros_like(image_3c)

    for i in range(height):
        i_start = max(i-half_psi_size, 0)
        i_end = min(i+half_psi_size, height) if is_psi_size_even else min(i+half_psi_size+1, height)

        for j in range(width):
            eroded = cv.erode(
                image_3c[
                    i_start : i_end, 
                    max(j-half_psi_size, 0) : min(j+half_psi_size, width) if is_psi_size_even else min(j+half_psi_size+1, width), 
                    :
                ], 
                cv.getStructuringElement(cv.MORPH_RECT, (omega_size, omega_size))
            )

            atm_light[i, j] = np.array([
                np.max(eroded[:, :, 0]),
                np.max(eroded[:, :, 1]),
                np.max(eroded[:, :, 2])
            ])

    print(f'Finished algorithm in process {process}, pid = {os.getpid()}, ps = {omega_size}')
    return atm_light


# Refine transmission map using guided image filter
def _refined_tmap(bgr_image: np.ndarray, base_tmap: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    # Ensure dtype "float32" to be compatible with cv.ximgproc.guidedFilter()
    gray_guidance = cv.cvtColor(bgr_image.astype(np.float32), cv.COLOR_BGR2GRAY)
    refined_tmap = cv.ximgproc.guidedFilter(gray_guidance, base_tmap.astype(np.float32), radius, epsilon)

    return refined_tmap.astype(DEFAULT_DTYPE)


def defoghaze(
    bgr_image: np.ndarray,
    patch_size: int = DEFAULT_PATCH_SIZE,
    omega: float = DEFAULT_OMEGA,   # control small amount of haze at distant objects
    t0: float = DEFAULT_T0,         # control lower bound of transmission map
    radius = DEFAULT_RADIUS,        # radius of guided filter
    epsilon = DEFAULT_EPS           # regularization term of guided filter
) -> dict:

    normalized_bgr = utils.minmax_normalize(bgr_image, new_dtype=DEFAULT_DTYPE)
    height, width, _ = normalized_bgr.shape
    dark_channel = _dark_channel(normalized_bgr, patch_size)

    min_edge = min(height, width)
    small_omega_size = max(int(min_edge/100*5), 1)
    big_omega_size = max(int(min_edge/100*15), 2)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        al_estimation_1 = executor.submit(_improved_atm_light, normalized_bgr, small_omega_size)
        al_estimation_2 = executor.submit(_improved_atm_light, normalized_bgr, big_omega_size)
    atm_light = (al_estimation_1.result() + al_estimation_2.result())/2

    # Estimate and refine transmission map
    base_tmap = 1 - omega * _dark_channel(normalized_bgr / atm_light, patch_size)
    refined_tmap = _refined_tmap(bgr_image, base_tmap, radius, epsilon)

    recovered_bgr = (normalized_bgr - atm_light) / np.maximum(refined_tmap.reshape(*refined_tmap.shape, 1), t0) + atm_light
    recovered_bgr = np.clip(recovered_bgr, 0, 1)

    return {
        'dark_channel': dark_channel,
        'atm_light': atm_light,
        'base_tmap': base_tmap,
        'refined_tmap': refined_tmap,
        'recovered_bgr': recovered_bgr
    }
