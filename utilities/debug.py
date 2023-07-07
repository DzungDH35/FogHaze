import math
import matplotlib.pyplot as plt
import numpy as np


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