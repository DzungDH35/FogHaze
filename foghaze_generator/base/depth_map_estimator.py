from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np


"""
@class A base class for depth map estimator which estimates (predicts) depth map from scene.
This estimator accepts multiple images (expected to be RGB) as input and produces depth maps (expected to be grayscale) as output.
A depth map can have two ways of representation:
    Direct: the closer a pixel is, the smaller its value, and the farther it is, the larger its value.
    Inverse: the closer a pixel is, the larger its value, the farther it is, the smaller its value.
"""
class BaseDepthMapEstimator(ABC):
    _rgb_images: list[np.ndarray] = []      # input as list of RGB images
    inverse_dmaps: list[np.ndarray] = []    # output as list of estimated inverse depth maps which are grayscale images


    def __init__(self, images: list[np.ndarray | str] = []):
        self.rgb_images = images
        

    @property
    def rgb_images(self) -> list[np.ndarray]:
        return self._rgb_images
    

    @rgb_images.setter
    def rgb_images(self, images: list[np.ndarray | str] = []):
        for i, img in enumerate(images):
            img_type = type(img)
            
            if img_type is str:
                file_path = img
                img = cv.imread(img)

                if img is None:
                    print(f'Cannot read the image file path: {file_path}!')
                    images.pop(i)
                else:
                    images[i] = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        self._rgb_images = images


    @abstractmethod
    def estimate_depth_maps(self) -> list[np.ndarray]:
        pass
