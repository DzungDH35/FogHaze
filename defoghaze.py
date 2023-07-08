from foghaze_removal.improved_dcp import DEFAULT_PATCH_SIZE, DEFAULT_OMEGA, DEFAULT_T0, DEFAULT_RADIUS, DEFAULT_EPS, defoghaze
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from utilities.debug import plot_multiple_images
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import utilities.utilities as utils


#     parser.add_argument('-ps', '--patch-size', type=int, default=DEFAULT_PATCH_SIZE, help='Local Patch Size')
#     parser.add_argument('-om', '--omega', type=float, default=DEFAULT_OMEGA, help='Omega controls small amount of haze at distant objects')
#     parser.add_argument('-t0', type=float, default=DEFAULT_T0, help='t0 controls lower bound of transmission map')
#     parser.add_argument('-r', '--radius', type=int, default=DEFAULT_RADIUS, help='Radius of guided filter')
#     parser.add_argument('-eps', '--epsilon', type=float, default=DEFAULT_EPS, help='Epsilon (regularization term of guided filter)')
#     parser.add_argument('-sm', '--save-mode', type=int, choices=[0, 1, 2], default=1, help='0 - not save, 1 - save only recovered image, 2 - save all related results')


RELATIVE_DIR_DFH_RESULT = '/defoghazing_output/'
FILE_SUFFIX_DARK_CHANNEL = '_dc'
FILE_SUFFIX_BASE_TMAP = '_base_tmap'
FILE_SUFFIX_REFINED_TMAP = '_refined_tmap'
FILE_SUFFIX_RECOVERED = '_recovered'
FILE_NAME_PERF_REPORT = 'performance_report.txt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path of a foggy/hazy image(s) or a directory of ones to be processed')
    parser.add_argument('-gp', '--gt-path', help='Path of corresponding ground-truth image or a directory of corresponding ones used for assessment')
    parser.add_argument('-op', '--output-path', help='Path of a directory to store defoghazing results')
    parser.add_argument('-sm', '--save-mode', type=int, choices=[0, 1, 2, 3], default=0, help='0 - no results are saved, 1 - save only defoghazing results, 2 - save only performance report, 3 - save all results')

    kwargs = vars(parser.parse_args())
    print('kwargs:', kwargs)

    input_path = kwargs.pop('input_path')
    bgr_images = utils.read_images_from_path(input_path)
    print('Num of BGR images:', len(bgr_images))
    bgr_gts = None
    if kwargs['gt_path']:
        bgr_gts = utils.read_images_from_path(kwargs.pop('gt_path'))
    
    output_path = kwargs.pop('output_path')
    save_mode = kwargs.pop('save_mode')

    # Run algorithm with performance measurement
    results = {}
    psnrs = {}
    ssims = {}
    for i, img in bgr_images.items():
        print('==================================================')
        print(f'Proccess image "{i}" with shape {img.shape} ({img.shape[0] * img.shape[1]} pixels not considering all 3 channels)', '\n')

        start = time.perf_counter()
        dfh_result = defoghaze(img)
        end= time.perf_counter()
        dfh_result['recovered_bgr'] = utils.minmax_normalize(dfh_result['recovered_bgr'], (0, 1), (0, 255), np.uint8)
        results[i] = dfh_result
        
        elapsed_time = end-start
        print('Speed (1-time measurement):')
        print('-- Execution time (s):', elapsed_time)
        print('-- FPS: ', 1/(elapsed_time), '\n')

        if bgr_gts:
            psnr = sk_psnr(bgr_gts[i], dfh_result['recovered_bgr'])
            ssim = sk_ssim(cv.cvtColor(bgr_gts[i], cv.COLOR_BGR2GRAY), cv.cvtColor(dfh_result['recovered_bgr'], cv.COLOR_BGR2GRAY))
            print('Image Quality:')
            print('-- PSNR: ', psnr)
            print('-- SSIM: ', ssim)
            psnrs[i] = psnr
            ssims[i] = ssim
        print('==================================================')
    avg_psnr = np.mean(list(psnrs.values()))
    avg_ssim = np.mean(list(ssims.values()))
    print('Average PSNR:', avg_psnr)
    print('Average SSIM:', avg_ssim)
    
    if save_mode:
        output_path = output_path or os.path.dirname(input_path) + RELATIVE_DIR_DFH_RESULT
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        
        if save_mode == 1 or save_mode == 3:
            for i, result in results.items():
                fname = os.path.splitext(i)[0]
                fname_dc = fname + FILE_SUFFIX_DARK_CHANNEL + '.jpg'
                fname_base_tmap = fname + FILE_SUFFIX_BASE_TMAP + '.jpg'
                fname_refined_tmap = fname + FILE_SUFFIX_REFINED_TMAP + '.jpg'
                fname_recovered = fname + FILE_SUFFIX_RECOVERED + '.jpg'

                plt.imsave(os.path.join(output_path, fname_dc), result['dark_channel'], cmap='gray')
                plt.imsave(os.path.join(output_path, fname_base_tmap), result['base_tmap'], cmap='gray')
                plt.imsave(os.path.join(output_path, fname_refined_tmap), result['refined_tmap'], cmap='gray')
                cv.imwrite(os.path.join(output_path, fname_recovered), result['recovered_bgr'])
            
        if save_mode == 2 or save_mode == 3:
            print(psnrs)
            print(avg_psnr)
            print(ssims)
            print(avg_ssim)
