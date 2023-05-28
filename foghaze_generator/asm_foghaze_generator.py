from .base.depth_map_estimator import BaseDepthMapEstimator
from .base.foghaze_generator import BaseFogHazeGenerator
import cv2 as cv
import numpy as np


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
    _depth_map_estimator: BaseDepthMapEstimator     # an estimator to predict depth map of scene
    _atm_light: int | np.ndarray[int]               # atmospheric light can be a constant or a pixel-position dependent value
    _scattering_coef: float | np.ndarray[float]     # scattering coefficient can be a constant or a pixel-position dependent value
    

    def __init__(
        self,
        dmap_estimator: BaseDepthMapEstimator,
        images: list[np.ndarray | str] = []
    ):
        super().__init__(images)

        if not isinstance(dmap_estimator, BaseDepthMapEstimator):
            raise TypeError('Depth map estimator must be of type BaseDepthMapEstimator!')
        
        self._depth_map_estimator = dmap_estimator

    
    @property
    def atm_light(self) -> int | np.ndarray[int]:
        return self._atm_light
    

    @atm_light.setter
    def atm_light(self, A: int | np.ndarray[int]):
        # Because [0 ,255] is the encoding value range of input images, then atmospheric light should also take value in this range.
        if (type(A) is int and (A < 0 or A > 255)) or (isinstance(A, np.ndarray) and np.any((A < 0) | (A > 255))):
            raise ValueError('Atmospheric light must be within [0, 255].')
        
        self._atm_light = A

    
    @property
    def scattering_coef(self) -> float | np.ndarray[float]:
        return self._scattering_coef
    

    @scattering_coef.setter
    def scattering_coef(self, beta: float | np.ndarray[float]):
        if (type(beta) is float and beta < 0) or (isinstance(beta, np.ndarray) and np.any(beta < 0)):
            raise ValueError('Scattering coefficient must be larger than 0.')

        self._scattering_coef = beta


    # @private
    def _generate_foghaze_image(self, original_img: np.ndarray) -> np.ndarray:
        atm_light = 255
        scattering_coef = 0.95

        self._depth_map_estimator.rgb_images = [original_img]
        depth_map = self._depth_map_estimator.estimate_depth_maps()[0]
        normalized_dmap = self._depth_map_estimator.normalize_depth_map(depth_map, True)
        normalized_dmap = cv.cvtColor(normalized_dmap, cv.COLOR_GRAY2RGB)
        normalized_dmap = normalized_dmap / 255
        
        transmission_map = np.exp(-scattering_coef * normalized_dmap)

        foghaze_img = original_img * transmission_map + atm_light * (1 - transmission_map)
        foghaze_img = np.array(foghaze_img, dtype=np.uint8)

        return foghaze_img


    def generate_foghaze_images(self) -> list[np.ndarray]:
        foghaze_images = []

        if len(self._rgb_images) == 0:
            print('There exists no input images!')
            return []

        for img in self._rgb_images:
            foghaze_images.append(self._generate_foghaze_image(img))
        
        return foghaze_images
