from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np


"""
@class A base class for depth map estimator which estimates (predicts) depth map from scene.
This estimator accepts multiple images (expected to be RGB) as input and produces depth maps (expected to be grayscale) as output.
"""
class BaseDepthMapEstimator(ABC):
    _rgb_images = [] # (np.ndarray[]) - list of RGB images


    # @param (mixture of np.ndarray and str) images
    def __init__(self, images=[]):
        self.rgb_images = images
        

    # @return np.ndarray[]
    @property
    def rgb_images(self):
        return self._rgb_images
    

    # @param (mixture of np.ndarray and str) images - expect images as numpy array or image file path
    @rgb_images.setter
    def rgb_images(self, images):
        if not isinstance(images, list):
            raise TypeError('Expected a list of images!')

        for i, img in enumerate(images):
            img_type = type(img)
            
            if img_type is str:
                img = cv.imread(img)

                if img is None:
                    print(f'Cannot read an image file path {img}!')
                    images.pop(i)
                else:
                    images[i] = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            elif not img_type is np.ndarray:
                raise TypeError('Not a valid image (numpy array) or string of file path!')
        
        self._rgb_images = images


    """
    @param (np.ndarray) dmap - depth map
    @param (bool) inverse - a depth map can have two ways of representation.
    First, the closer a pixel is, the smaller its value, and the farther it is, the larger its value.
    Second,the closer a pixel is, the larger its value, the farther it is, the smaller its value.
    Set inverse = True will switch back and forth.

    @return np.ndarray
    """
    @abstractmethod
    def normalize_depth_map(dmap, inverse=False):
        pass


    # @return (np.ndarray[]) - list of grayscale depth maps
    @abstractmethod
    def estimate_depth_maps(self):
        pass
