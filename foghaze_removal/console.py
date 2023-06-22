from DCP import DEFAULT_PATCH_SIZE, DEFAULT_OMEGA, DEFAULT_T0, DEFAULT_RADIUS, DEFAULT_EPS, defoghaze
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time
import math


def print_img_info(img, print_img: bool = False):
    print(
        'Image information:',
        f'(height, width, channel): {img.shape}',
        f'Dtype: {img.dtype}',
        f'Min value: {np.amin(img)}',
        f'Max value: {np.amax(img)}',
        sep='\n')
    
    if print_img:
        print(img)
    print()


def plot_multiple_images(images, cmap='gray'):
    num_imgs = len(images)

    if num_imgs == 1:
        plt.imshow(images[0], cmap)
        plt.axis('off')
    elif num_imgs == 2:
        _, axs = plt.subplots(1, 2)
        axs[0].imshow(images[0], cmap=cmap)
        axs[0].axis('off')
        axs[1].imshow(images[1], cmap=cmap)
        axs[1].axis('off')
    else:
        num_rows = math.ceil(num_imgs / 2)
        _, axs = plt.subplots(num_rows, 2)
        for i in range(num_rows):
            axs[i][0].imshow(images[i*2], cmap=cmap)
            axs[i][0].axis('off')
            axs[i][1].imshow(images[i*2+1], cmap=cmap)
            axs[i][1].axis('off')
        
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', help='Input Image Path')
    parser.add_argument('-ps', '--patch-size', type=int, default=DEFAULT_PATCH_SIZE, help='Local Patch Size')
    parser.add_argument('-om', '--omega', type=float, default=DEFAULT_OMEGA, help='Omega controls small amount of haze at distant objects')
    parser.add_argument('-t0', type=float, default=DEFAULT_T0, help='t0 controls lower bound of transmission map')
    parser.add_argument('-r', '--radius', type=int, default=DEFAULT_RADIUS, help='Radius of guided filter')
    parser.add_argument('-eps', '--epsilon', type=float, default=DEFAULT_EPS, help='Epsilon (regularization term of guided filter)')

    kwargs = vars(parser.parse_args())

    input_image_path = kwargs.pop('input_image')
    bgr_image = cv.imread(input_image_path)
    if bgr_image is None:
        raise Exception(f'Cannot read the image file path: "{input_image_path}"')
    
    result = defoghaze(bgr_image, **kwargs)

    rgb = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
    recovered_rgb = cv.cvtColor(result['recovered_bgr'], cv.COLOR_BGR2RGB)
    plot_multiple_images([rgb, recovered_rgb, result['dark_channel'], result['base_tmap']])
