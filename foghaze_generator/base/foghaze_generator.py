from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np


"""
@class A base class for fog-haze generator which accepts multiple images as input and produces corresponding foggy-hazy images as output.
This generator expects input and output images as RGB.
"""
class BaseFogHazeGenerator(ABC):
    _rgb_images: list[np.ndarray] = []  # list of original RGB images
    fh_images: list[np.ndarray] = []    # list of RGB foggy-hazy images


    def __init__(self, images: list[np.ndarray | str] = []):
        self.rgb_images = images


    @property
    def rgb_images(self) -> np.ndarray:
        return self._rgb_images


    @rgb_images.setter
    def rgb_images(self, images: list[np.ndarray | str]):
        print('parent setter')
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
    def generate_foghaze_images(self) -> list[np.ndarray]:
        pass
