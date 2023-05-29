from .base.depth_map_estimator import BaseDepthMapEstimator
from .base.foghaze_generator import BaseFogHazeGenerator
import cv2 as cv
import numpy as np
import random


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
    _depth_maps: list[np.ndarray] = []                      # list of inverse relative depth maps which are grayscale images
    _atm_lights: list[int | np.ndarray[int]] = []           # list of atmospheric lights, each can be a constant or a pixel-position dependent value
    _scattering_coefs: list[float | np.ndarray[float]] = [] # list of scattering coefficients, each can be a constant or a pixel-position dependent value


    def __init__(
        self,
        dmap_estimator: BaseDepthMapEstimator,
        images: list[np.ndarray | str] = [],
        depth_maps: list[np.ndarray | str] = [],
        atm_lights: list[int | np.ndarray[int]] = [],
        betas: list[float | np.ndarray[float]] = []
    ):
        super().__init__(images)

        if not isinstance(dmap_estimator, BaseDepthMapEstimator):
            raise TypeError('Depth map estimator must be of type BaseDepthMapEstimator!')
        
        self._depth_map_estimator = dmap_estimator
        self.depth_maps = depth_maps
        self.atm_lights = atm_lights
        self.scattering_coefs = betas
    

    @property
    def depth_maps(self) -> list[np.ndarray]:
        return self._depth_maps
    

    @depth_maps.setter
    def _depth_maps(self, dmaps: list[np.ndarray | str]):
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
        
        self._depth_maps = dmaps

    
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


    # @private
    def _generate_foghaze_image(
        self,
        original_img: np.ndarray,
        dmap: np.ndarray,
        atm_light: int | np.ndarray[int],
        scattering_coef: float | np.ndarray[float]
    ) -> tuple:
        
        if dmap is None:
            self._depth_map_estimator.rgb_images = [original_img]
            dmap = self._depth_map_estimator.estimate_depth_maps()[0]
        
        if atm_light is None:
            atm_light = random.randint(0, 255)
        
        if scattering_coef is None:
            scattering_coef = random.random()

        normalized_dmap = self._depth_map_estimator.normalize_depth_map(dmap, True)
        normalized_dmap /= 255
        normalized_dmap = cv.cvtColor(normalized_dmap, cv.COLOR_GRAY2RGB)
        
        transmission_map = np.exp(-scattering_coef * normalized_dmap)

        foghaze_img = original_img * transmission_map + atm_light * (1 - transmission_map)
        foghaze_img = np.array(foghaze_img, dtype=np.uint8)

        return (foghaze_img, dmap, atm_light, scattering_coef)


    def generate_foghaze_images(self) -> list[np.ndarray]:
        self.fh_images = []

        if len(self._rgb_images) == 0:
            print('There exists no input images!')
            return []

        for i, img in enumerate(self._rgb_images):
            result = self._generate_foghaze_image(
                img,
                self._depth_maps[i],
                self._atm_lights[i],
                self._scattering_coefs[i]
            )
            
            self.fh_images[i] = result[0]
            self._depth_maps[i] = result[1]
            self._atm_lights[i] = result[2]
            self._scattering_coefs[i] = result[3]
        
        return self.fh_images
