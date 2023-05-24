from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np


"""
@class A base class for depth map estimator which estimates (predicts) depth map from scene.
This estimator accepts multiple images as input and produces depth maps as output.
"""
class BaseDepthMapEstimator(ABC):
    _base_images = [] # (np.ndarray[]) - list of base images which are RGB (expected)


    # @param (mixture of np.ndarray and str) images
    def __init__(self, images=[]):
        self.base_images = images
        

    # @return np.ndarray[]
    @property
    def base_images(self):
        return self._base_images
    

    # @param (mixture of np.ndarray and str) images - expect images as numpy array or image file path
    @base_images.setter
    def base_images(self, images):
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
        
        self._base_images = images


    # @return np.ndarray[]
    @abstractmethod
    def estimate_depth_maps(self):
        pass
