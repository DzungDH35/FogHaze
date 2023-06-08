from base.depth_map_estimator import BaseDepthMapEstimator
from base.foghaze_generator import BaseFogHazeGenerator
from helper import get_perlin_noise
import cv2 as cv
import numpy as np
import random


HUGE_NUMBER = 999999
ATM_LIGHT_BOUNDS = (0, 255)                 # 0 <= atmospheric light <= 255
SCATTERING_COEF_BOUNDS = (0, HUGE_NUMBER)   # 0 <= scattering coefficient <= infinity


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
    _depth_map_estimator: BaseDepthMapEstimator                         # @private An estimator to predict depth map of scene.
    _inverse_dmaps: list[np.ndarray]                                    # @private inverse relative depth map (grayscale)
    _atm_lights: list[int | np.ndarray[int] | tuple[int]]               # @private atmospheric light (value | pixel-dependent | tuple as range)
    _scattering_coefs: list[float | np.ndarray[float] | tuple[float]]   # @private scattering coefficients (value | pixel-dependent | tuple as range)

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


    """
    'atm_light_estimation': 'naive_int' | 'naive_arr' - currently, only naive algorithm is available
    'scattering_coef_estimation': 'naive_float' | 'naive_arr' | 'pnoise'
    """
    operation_mode = {
        'atm_light': 'naive_int',
        'scattering_coef': 'pnoise'
    }


    def __init__(
        self,
        dmap_estimator: BaseDepthMapEstimator,
        images: list[np.ndarray | str] = [],
        inverse_dmaps: list[np.ndarray | str] = [],
        atm_lights: list[int | np.ndarray[int] | tuple[int]] = [],
        betas: list[float | np.ndarray[float] | tuple[float]] = [],
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
    

    # @override
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
    def atm_lights(self) -> list[int | np.ndarray[int] | tuple[int]]:
        return self._atm_lights
    

    @atm_lights.setter
    def atm_lights(self, atm_lights: list[int | np.ndarray[int] | tuple[int]]):
        low, high = ATM_LIGHT_BOUNDS

        for i, A in enumerate(atm_lights):
            if (type(A) is int and (A < low or A > high)) or (isinstance(A, np.ndarray) and np.any(A < low | A > high)) or (type(A) is tuple and (A[0] < low | A[1] > high)):
                raise ValueError(f'Atmospheric light at index {i} must be within [{low}, {high}].')
        
        size_diff = len(self._rgb_images) - len(atm_lights)
        if size_diff > 0:
            atm_lights += [None] * size_diff

        self._atm_lights = atm_lights

    
    @property
    def scattering_coefs(self) -> list[float | np.ndarray[float] | tuple[float]]:
        return self._scattering_coefs
    

    @scattering_coefs.setter
    def scattering_coefs(self, betas: list[float | np.ndarray[float] | tuple[float]]):
        low, high = SCATTERING_COEF_BOUNDS

        for i, beta in enumerate(betas):
            if (type(beta) is float and (beta < low or beta > high)) or (isinstance(beta, np.ndarray) and np.any(beta < low | beta > high)) or (type(beta) is tuple and (beta[0] < low | beta[1] > high)):
                raise ValueError(f'Scattering coefficient at index {i} must be within [{low}, {high}].')

        size_diff = len(self._rgb_images) - len(betas)
        if size_diff > 0:
            betas += [None] * size_diff

        self._scattering_coefs = betas


    @property
    def pnoise_configs(self) -> list[dict]:
        return self._pnoise_configs


    @pnoise_configs.setter
    def pnoise_configs(self, configs: list[dict]):
        size_diff = len(self._rgb_images) - len(configs)
        if size_diff > 0:
            configs += [None] * size_diff
        self._pnoise_configs = configs


    # @private Reset configurations of the generator
    def _reset_configs(self):
        self.inverse_dmaps = []
        self.atm_lights = []
        self.scattering_coefs = []
        self.pnoise_configs = []


    # @private Randomize a value or a numpy array of atmospheric light.
    def _rand_atm_light(self, np_shape: tuple = None, range: tuple = None) -> int | np.ndarray[int]:
        if range:
            low, high = range
        else:
            low, high = ATM_LIGHT_BOUNDS

        if not np_shape:
            return random.randint(low, high)
        
        height, width, channel = np_shape
        arr_2d = np.random.randint(low, high+1, size=(height, width))
        arr_3d = np.repeat(arr_2d[:, :, np.newaxis], channel, axis=2)

        return arr_3d

    
    # @private Randomize a value or a numpy array of scattering coefficients.
    def _rand_scattering_coef(self, np_shape: tuple = None, range: tuple = None) -> float | np.ndarray[float]:
        if range:
            low, high = range
        else:
            low, high = SCATTERING_COEF_BOUNDS

        if not np_shape:
            return random.uniform(low, high)
        
        height, width, channel = np_shape
        arr_2d = np.random.uniform(low, high, size=(height, width))
        arr_3d = np.repeat(arr_2d[:, :, np.newaxis], channel, axis=2)

        return arr_3d


    # @private Generate scattering coefficient using Perlin noise.
    def _gen_perlin_scattering_coef(self, np_shape: tuple, pnoise_config: dict = {}, range: tuple = None):
        if range:
            value_bounds = range
        else:
            value_bounds = SCATTERING_COEF_BOUNDS
        return get_perlin_noise(np_shape, pnoise_config, value_bounds)


    # @private
    def _generate_foghaze_image(self, img_idx: int) -> tuple:
        clear_img = self.rgb_images[img_idx]
        img_shape = clear_img.shape
        inverse_dmap = self.inverse_dmaps[img_idx]
        atm_light = self.atm_lights[img_idx]
        scattering_coef = self.scattering_coefs[img_idx]
        
        if inverse_dmap is None:
            self._depth_map_estimator.rgb_images = [clear_img]
            inverse_dmap = self._depth_map_estimator.estimate_depth_maps()[0]
        
        if atm_light is None or type(atm_light) is tuple:
            bounds = atm_light if atm_light else None

            if self.operation_mode['atm_light'] == 'naive_int':
                atm_light = self._rand_atm_light(range=bounds)
            else:
                atm_light = self._rand_atm_light(img_shape, bounds)
        
        if scattering_coef is None or type(scattering_coef) is tuple:
            bounds = scattering_coef if scattering_coef else None
            opmode = self.operation_mode['scattering_coef'] 
            
            if opmode == 'naive_float':
                scattering_coef = self._rand_scattering_coef(range=bounds)
            elif opmode == 'naive_arr':
                scattering_coef = self._rand_scattering_coef(img_shape, bounds)
            else:
                scattering_coef = self._gen_perlin_scattering_coef(img_shape, self.pnoise_configs[img_idx], bounds)
        
        normalized_idmap = cv.normalize(inverse_dmap, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F) # scale to [0.0 - 1.0]
        normalized_idmap = 1.0 - normalized_idmap # reverse the inverse depth map
        normalized_idmap = cv.cvtColor(normalized_idmap, cv.COLOR_GRAY2RGB)
        
        transmission_map = np.exp(-scattering_coef * normalized_idmap)

        foghaze_img = clear_img * transmission_map + atm_light * (1 - transmission_map)
        foghaze_img = np.array(foghaze_img, dtype=np.uint8)

        return (foghaze_img, inverse_dmap, atm_light, scattering_coef)


    # @override
    def generate_foghaze_images(self) -> list[np.ndarray]:
        self.fh_images = []

        if len(self._rgb_images) == 0:
            print('There exists no input images!')
            return []

        for i, img in enumerate(self._rgb_images):
            result = self._generate_foghaze_image(i)

            self.fh_images.append(result[0])
            self._inverse_dmaps[i] = result[1]
            self._atm_lights[i] = result[2]
            self._scattering_coefs[i] = result[3]
        
        return self.fh_images
