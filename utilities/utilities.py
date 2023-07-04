import cv2 as cv
import os


VALID_IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

"""
Path, which is of an image or a directory of images, is read into a list of RGB or BGR images.
color_mode is BGR (default) or RGB
"""
def read_images_from_path(path: str, color_mode='BGR') -> list:
    images = []

    if os.path.isfile(path) and os.path.splitext(path)[1].lower() in VALID_IMG_EXTENSIONS:
        img = cv.imread(path)

        if img is None:
            raise Exception(f'Image path "{path}" cannot be read!')
        images.append(cv.imread(path))
    elif os.path.isdir(path):
        for file_name in os.listdir(path):

            if os.path.splitext(file_name)[1].lower() in VALID_IMG_EXTENSIONS:
                img = cv.imread(os.path.join(path, file_name))

                if img is None:
                    raise Exception(f'Image path "{path}" cannot be read!')
                images.append(img)
    else:
        raise Exception(f'Path "{path}" is not found in file system!')
    
    if color_mode == 'RGB':
        images = [cv.cvtColor(img, cv.COLOR_BGR2RGB) for img in images]

    return images
