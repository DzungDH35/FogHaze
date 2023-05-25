from .base.depth_map_estimator import BaseDepthMapEstimator
from .base.foghaze_generator import BaseFogHazeGenerator
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
    _depth_map_estimator = None # (BaseDepthMapEstimator) - estimate depth map of scene
    

    """
    @param (mixture of np.ndarray or str) images
    @param (BaseDepthMapEstimator) dmap_estimator
    """
    def __init__(self, images, dmap_estimator):
        super().__init__(images)

        if not isinstance(dmap_estimator, BaseDepthMapEstimator):
            raise Exception('Depth map estimator is not initialized properly!')
        
        self._depth_map_estimator = dmap_estimator


    """
    @private
    @param (np.ndarray) original_img
    @return np.ndarray
    """
    def _generate_foghaze_image(self, original_img):
        atm_light = 255
        scattering_coef = 0.95

        self._depth_map_estimator.rgb_images = [original_img]
        depth_map = self._depth_map_estimator.estimate_depth_maps()[0]
        depth_map_3c = np.zeros_like(original_img)
        depth_map_3c[:, :, 0] = depth_map
        depth_map_3c[:, :, 1] = depth_map
        depth_map_3c[:, :, 2] = depth_map
        normalized_dmap = depth_map_3c / 255
        
        transmission_map = np.exp(-scattering_coef * normalized_dmap)

        foghaze_img = original_img * transmission_map + atm_light * (1 - transmission_map)
        foghaze_img = np.array(foghaze_img, dtype=np.uint8)

        return foghaze_img


    # @return np.ndarray
    def generate_foghaze_images(self):
        foghaze_images = []

        if len(self._rgb_images) == 0:
            print('There exists no input images!')
            return []

        for img in self._rgb_images:
            foghaze_images.append(self._generate_foghaze_image(img))
        
        return foghaze_images
