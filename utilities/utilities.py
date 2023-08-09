import cv2 as cv
import numpy as np
import os


VALID_IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


# Use minmax normalization to scale range of an array to a new range (default is [0, 1])
def minmax_normalize(arr: np.ndarray, current_range: tuple = None, new_range: tuple = (0, 1), new_dtype = None) -> np.ndarray:
    if not current_range:
        if arr.dtype == np.uint8:
            current_range = (0, 255)
        elif arr.dtype == np.uint16:
            current_range = (0, 65535)
        else:
            raise TypeError('Must provide current range of values!')

    if not new_dtype:
        new_dtype = arr.dtype
    
    if current_range != new_range:
        low_current, high_current = current_range
        low_new, high_new = new_range

        new_arr = low_new + (arr.astype(np.float64) - low_current) * (high_new - low_new) / (high_current - low_current)

    return new_arr.astype(new_dtype)


"""
Path, which is of an image or a directory of images, is read into a list or dict (if indexed) of color or gray images.
color_mode is BGR (default) or RGB or gray.
If index = True, a dict indexed with filename.ext will be returned, otherwise, a list returned.
If a list is returned, it is the result of images read in alphabetical order.
"""
def read_images_from_path(path: str, color_mode='BGR', index: bool = True):
    file_names = []
    images = []
    cv_imread_mode = cv.IMREAD_GRAYSCALE if color_mode == 'gray' else cv.IMREAD_COLOR

    if os.path.isfile(path) and os.path.splitext(path)[1].lower() in VALID_IMG_EXTENSIONS:
        img = cv.imread(path, cv_imread_mode)

        if img is None:
            raise Exception(f'Image path "{path}" cannot be read!')
        
        file_names.append(os.path.basename(path))
        images.append(cv.imread(path))
    elif os.path.isdir(path):
        for file_name in sorted(os.listdir(path)):

            if os.path.splitext(file_name)[1].lower() in VALID_IMG_EXTENSIONS:
                img = cv.imread(os.path.join(path, file_name), cv_imread_mode)

                if img is None:
                    raise Exception(f'Image path "{path}" cannot be read!')
                
                file_names.append(file_name)
                images.append(img)
    else:
        raise Exception(f'Path "{path}" is not found in file system!')

    if color_mode == 'RGB':
        images = [cv.cvtColor(img, cv.COLOR_BGR2RGB) for img in images]

    if index is True:
        return {index: content for index, content in zip(file_names, images)}

    return images
