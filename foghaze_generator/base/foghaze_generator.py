from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np


class BaseFogHazeGenerator(ABC):
    _original_images = [] # (np.ndarray[]) - list of original image which is RGB (expected)


    # @param (mixture of np.ndarray and str) images
    def __init__(self, images):
        self.original_images = images


    # @return np.ndarray
    @property
    def original_images(self):
        return self._original_images


    @original_images.setter
    def original_images(self, images):
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
        
        self._original_images = images

    
    # @return np.ndarray[]
    @abstractmethod
    def generate_foghaze_images(self):
        pass
