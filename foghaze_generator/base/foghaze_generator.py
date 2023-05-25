from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np


"""
@class A base class for fog-haze generator which accepts multiple images as input and produces corresponding foggy-hazy images as output.
This generator expects input and output images as RGB.
"""
class BaseFogHazeGenerator(ABC):
    _rgb_images = [] # (np.ndarray[]) - list of original RGB images


    # @param (mixture of np.ndarray or str) images
    def __init__(self, images=[]):
        self.rgb_images = images


    # @return np.ndarray
    @property
    def rgb_images(self):
        return self._rgb_images


    # @param (mixture of np.ndarray or str) images - original images which can be numpy array (expected to be RGB) or file path
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

    
    # @return (np.ndarray[]) - list of RGB foggy-hazy images
    @abstractmethod
    def generate_foghaze_images(self):
        pass
