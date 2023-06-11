from noise import pnoise2
import numpy as np


# Scale an array from range [a, b] to [c, d]
def scale_array(arr: np.ndarray, old_range: tuple, new_range: tuple) -> np.ndarray:
    low_old, high_old = old_range
    low_new, high_new = new_range

    return low_new + (arr - low_old) * (high_new - low_new) / (high_old - low_old)


# Generate Perlin noise as a 3-channel numpy array, each value is a float within [-1, 1] by default.
def get_perlin_noise(np_shape: tuple, pnoise_config: dict = {}, scaled_range: tuple = None) -> np.ndarray[float]:
    height, width, channel = np_shape
    noise = np.zeros((height, width))
    pnoise_config = pnoise_config.copy()
    scale = pnoise_config.pop('scale') if 'scale' in pnoise_config else 1

    for y in range(height):
        for x in range(width):
            noise[y, x] = pnoise2(x*scale, y*scale, **pnoise_config)

    if scaled_range:
        noise = scale_array(noise, (-1, 1), scaled_range)

    noise = np.repeat(noise[:, :, np.newaxis], channel, axis=2)

    return noise
