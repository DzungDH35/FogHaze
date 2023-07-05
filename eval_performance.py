from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim
import argparse
import cv2 as cv
import numpy as np
from utilities.utilities import read_images_from_path


def psnr(gt_set, recovered_set) -> tuple[list, float]:
    psnrs = []
    num_images = len(gt_set)
    
    for i in range(num_images):
        psnrs.append(_psnr(gt_set[i], recovered_set[i]))
    
    return (psnrs, np.mean(psnrs))


def ssim(gt_set, recovered_set) -> tuple[list, float]:
    ssims = []
    num_images = len(gt_set)
    
    for i in range(num_images):
        ssims.append(_ssim(
            cv.cvtColor(gt_set[i], cv.COLOR_BGR2GRAY),
            cv.cvtColor(recovered_set[i], cv.COLOR_BGR2GRAY),
        ))
    
    return (ssims, np.mean(ssims))


if __name__ == '__main__':
    bgr_gts = []
    bgr_recovered = []

    parser = argparse.ArgumentParser()
    parser.add_argument('ground-truth-path', help='Path of a ground-truth image or a directory of ground-truth images')
    parser.add_argument('recovery-path', help='Path of a recovered image or a directory of recovered images')

    kwargs = vars(parser.parse_args())
    bgr_gts = read_images_from_path(kwargs['ground-truth-path'])
    bgr_recovered = read_images_from_path(kwargs['recovery-path'])
    print(len(bgr_gts))
    print(len(bgr_recovered))

    if len(bgr_gts) != len(bgr_recovered):
        raise Exception('Ground-truth data set and Recover data set are not equal in size!')
    
    eval_result = {
        'psnr': psnr(bgr_gts, bgr_recovered),
        'ssim': ssim(bgr_gts, bgr_recovered)
    }

    print(eval_result)
    # save result ...
