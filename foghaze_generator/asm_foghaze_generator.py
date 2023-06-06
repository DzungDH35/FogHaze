from .base.depth_map_estimator import BaseDepthMapEstimator
from .base.foghaze_generator import BaseFogHazeGenerator
from noise import pnoise2
import cv2 as cv
import numpy as np
import random


ATM_LIGHT_BOUNDS = (0, 255) # 0 <= atmospheric light <= 255
SCATTERING_COEF_BOUNDS = (0, 3) # 0 <= scattering coefficient <= 3


"""
@class An implementation of fog-haze generator which utilizes atmospheric scattering model.

Atmospheric Scattering Model (ASM): 
    I(x) = J(x) * t(x) + A(1 - t(x)), in which:

    I(x): hazy image
    J(x): haze-free image
    t(x): transmission map
    A: atmospheric light

    t(x) = e^(-beta * d(x)), in which:
    d(x): depth map
"""
class ASMFogHazeGenerator(BaseFogHazeGenerator):
    _depth_map_estimator: BaseDepthMapEstimator = None      # an estimator to predict depth map of scene
    _inverse_dmaps: list[np.ndarray] = []                   # list of inverse relative depth maps which are grayscale images
    _atm_lights: list[int | np.ndarray[int]] = []           # list of atmospheric lights, each can be a constant or a pixel-position dependent value
    _scattering_coefs: list[float | np.ndarray[float]] = [] # list of scattering coefficients, each can be a constant or a pixel-position dependent value

    """
    @private List of Perlin noise configurations (each for a corresponding input image).
    Each configuration is a dict with the following structure:
    {
        'base': int             # seed to generate different patterns
        'lacunarity': float     # control the frequencies of the octaves
        'octaves': int          # number of octaves, each of which is a Perlin noise, used to control the level of details
        'persistence': float    # control the amplitudes of the octaves, hence, control the roughness
        'repeatx': int          # control the repeat of pattern along the x-axis
        'repeaty': int          # control the repeat of pattern along the y-axis
        'scale': float          # scale of the net noise (sum of octaves)
    }
    """
    _pnoise_configs: list[dict]


    def __init__(
        self,
        dmap_estimator: BaseDepthMapEstimator,
        images: list[np.ndarray | str] = [],
        inverse_dmaps: list[np.ndarray | str] = [],
        atm_lights: list[int | np.ndarray[int]] = [],
        betas: list[float | np.ndarray[float]] = [],
        pnoise_configs: list[dict] = []
    ):
        super().__init__(images)

        if not isinstance(dmap_estimator, BaseDepthMapEstimator):
            raise TypeError('Depth map estimator must be of type BaseDepthMapEstimator!')
        
        self._depth_map_estimator = dmap_estimator
        self.inverse_dmaps = inverse_dmaps
        self.atm_lights = atm_lights
        self.scattering_coefs = betas
        self.pnoise_configs = pnoise_configs
    

    @BaseFogHazeGenerator.rgb_images.setter
    def rgb_images(self, images: list[np.ndarray | str]):
        super(ASMFogHazeGenerator, ASMFogHazeGenerator).rgb_images.__set__(self, images)
        self._reset_configs()


    @property
    def inverse_dmaps(self) -> list[np.ndarray]:
        return self._inverse_dmaps
    

    @inverse_dmaps.setter
    def inverse_dmaps(self, dmaps: list[np.ndarray | str]):
        for i, dmap in enumerate(dmaps):
            img_type = type(dmap)
            
            if img_type is str:
                file_path = dmap
                dmap = cv.imread(dmap)

                if dmap is None:
                    print(f'Cannot read the image file path: {file_path}!')
                    dmaps.pop(i)
                else:
                    dmaps[i] = cv.cvtColor(dmap, cv.COLOR_BGR2GRAY)
        
        size_diff = len(self._rgb_images) - len(dmaps)
        if size_diff > 0:
            dmaps += [None] * size_diff
        
        self._inverse_dmaps = dmaps

    
    @property
    def atm_lights(self) -> list[int | np.ndarray[int]]:
        return self._atm_lights
    

    @atm_lights.setter
    def atm_lights(self, atm_lights: list[int | np.ndarray[int]]):
        for i, A in enumerate(atm_lights):
            # Because [0 ,255] is the encoding value range of input images, then atmospheric light should also take value in this range.
            if (type(A) is int and (A < 0 or A > 255)) or (isinstance(A, np.ndarray) and np.any((A < 0) | (A > 255))):
                raise ValueError(f'Atmospheric light at index {i} must be within [0, 255].')
        
        size_diff = len(self._rgb_images) - len(atm_lights)
        if size_diff > 0:
            atm_lights += [None] * size_diff

        self._atm_lights = atm_lights

    
    @property
    def scattering_coefs(self) -> list[float | np.ndarray[float]]:
        return self._scattering_coefs
    

    @scattering_coefs.setter
    def scattering_coefs(self, betas: list[float | np.ndarray[float]]):
        for i, beta in enumerate(betas):
            if (type(beta) is float and beta < 0) or (isinstance(beta, np.ndarray) and np.any(beta < 0)):
                raise ValueError(f'Scattering coefficient at index {i} must be larger than 0.')

        size_diff = len(self._rgb_images) - len(betas)
        if size_diff > 0:
            betas += [None] * size_diff

        self._scattering_coefs = betas


    @property
    def pnoise_configs(self) -> list[dict]:
        return self._pnoise_configs


    @pnoise_configs.setter
    def pnoise_configs(self, configs: list[dict]):
        self._pnoise_configs = configs


    # @private Reset configurations of the generator
    def _reset_configs(self):
        self.inverse_dmaps = []
        self.atm_lights = []
        self.scattering_coefs = []
        self.pnoise_configs = []


    # @private Scale an array from range [a, b] to [c, d]
    def _scale_array(self, arr: np.ndarray, old_range: tuple, new_range: tuple) -> np.ndarray:
        low_old, high_old = old_range
        low_new, high_new = new_range

        return low_new + (arr - low_old) * (high_new - low_new) / (high_old - low_old)


    """
    @private Randomize a constant or a numpy array of (int) atmospheric light.
    """
    def _rand_atm_light(self, np_shape: tuple = None) -> int | np.ndarray[int]:
        low, high = ATM_LIGHT_BOUNDS

        if not np_shape:
            return random.randint(low, high)
        
        height, width, channel = np_shape
        arr_2d = np.random.randint(low, high+1, size=(height, width))
        arr_3d = np.repeat(arr_2d[:, :, np.newaxis], channel, axis=2)

        return arr_3d

    
    """
    @private Randomize a constant or a numpy array of (float) scattering coefficients.
    """
    def _rand_scattering_coef(self, np_shape: tuple = None) -> float | np.ndarray[float]:
        low, high = SCATTERING_COEF_BOUNDS

        if not np_shape:
            return random.uniform(low, high)
        
        height, width, channel = np_shape
        arr_2d = np.random.uniform(low, high, size=(height, width))
        arr_3d = np.repeat(arr_2d[:, :, np.newaxis], channel, axis=2)

        return arr_3d


    # @private Generate Perlin noise as a 3-channel numpy array, each value is a float within [-1, 1].
    def _get_perlin_noise(np_shape: tuple, pnoise_config: dict = {}) -> np.ndarray[float]:
        height, width, channel = np_shape
        noise = np.zeros((height, width))
        scale = pnoise_config.pop('scale') if pnoise_config.get('scale') else 1

        for y in range(height):
            for x in range(width):
                noise[y, x] = pnoise2(x*scale, y*scale, **pnoise_config)

        noise = np.repeat(noise[:, :, np.newaxis], channel, axis=2)

        return noise

    
    # @private
    def _generate_foghaze_image(
        self,
        original_img: np.ndarray,
        inverse_dmap: np.ndarray,
        atm_light: int | np.ndarray[int],
        scattering_coef: float | np.ndarray[float]
    ) -> tuple:
        
        if inverse_dmap is None:
            self._depth_map_estimator.rgb_images = [original_img]
            inverse_dmap = self._depth_map_estimator.estimate_depth_maps()[0]
        
        if atm_light is None:
            atm_light = random.randint(0, 255)
        
        if scattering_coef is None:
            scattering_coef = random.uniform(0, 2)

        normalized_dmap = cv.normalize(inverse_dmap, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F) # scale to [0.0 - 1.0]
        normalized_dmap = 1.0 - normalized_dmap # reverse the inverse depth map
        normalized_dmap = cv.cvtColor(normalized_dmap, cv.COLOR_GRAY2RGB)
        
        transmission_map = np.exp(-scattering_coef * normalized_dmap)

        foghaze_img = original_img * transmission_map + atm_light * (1 - transmission_map)
        foghaze_img = np.array(foghaze_img, dtype=np.uint8)

        return (foghaze_img, inverse_dmap, atm_light, scattering_coef)


    def generate_foghaze_images(self) -> list[np.ndarray]:
        self.fh_images = []

        if len(self._rgb_images) == 0:
            print('There exists no input images!')
            return []

        for i, img in enumerate(self._rgb_images):
            result = self._generate_foghaze_image(
                img,
                self._inverse_dmaps[i],
                self._atm_lights[i],
                self._scattering_coefs[i]
            )

            self.fh_images.append(result[0])
            self._inverse_dmaps[i] = result[1]
            self._atm_lights[i] = result[2]
            self._scattering_coefs[i] = result[3]
        
        return self.fh_images
