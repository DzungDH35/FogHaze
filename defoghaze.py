from foghaze_removal.dcp import defoghaze as dcp_defoghaze
from foghaze_removal.improved_dcp import defoghaze as idcp_defoghaze
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


DCP_PARSER_CONFIG = {
    'patch-size': {
        'short': 'ps',
        'type': int,
        'required': False,
        'help': 'Path of a foggy/hazy image(s) or a directory of ones to be processed'
    },
    'omega': {
        'short': 'om',
        'type': float,
        'required': False,
        'help': 'Omega controls small amount of haze at distant objects'
    },
    't0': {
        'type': float,
        'required': False,
        'help': 't0 controls lower bound of transmission map'
    },
    'radius': {
        'short': 'r',
        'type': int,
        'required': False,
        'help': 'Radius of guided filter'
    },
    'epsilon': {
        'short': 'eps',
        'type': float,
        'required': False,
        'help': 'Epsilon (regularization term of guided filter)'
    }
}
IMPROVED_DCP_PARSER_CONFIG = {
    **DCP_PARSER_CONFIG,
    'fusion_weight': {
        'short': 'fw',
        'type': float,
        'required': False,
        'help': 'Fusion weight is used to blend transmission maps obtained from multi-scale analysis'
    }
}

SUPPORTED_ALGORITHMS = ('dcp', 'improved_dcp')

RELATIVE_DIR_DFH_RESULT = '/defoghazing_output/'
FILE_SUFFIX_DARK_CHANNEL = '_dc'
FILE_SUFFIX_BASE_TMAP = '_base_tmap'
FILE_SUFFIX_REFINED_TMAP = '_refined_tmap'
FILE_SUFFIX_RECOVERED = '_recovered'
FILE_NAME_PERF_REPORT = 'performance_report.txt'


def add_algorithm_arguments(parser: argparse.ArgumentParser, parser_config: dict):
    for arg_key, config in parser_config.items():
        parser.add_argument(
            f'-{config.pop("short", arg_key)}',
            f'--{arg_key}',
            **config
        )


if __name__ == '__main__':
    print('Supported algorithms:', SUPPORTED_ALGORITHMS)
    algo = input('Select algorithm: ')
    if algo not in SUPPORTED_ALGORITHMS:
        print('Not supported algorithm!')

    if algo == 'dcp':
        algo_parser_config = DCP_PARSER_CONFIG
        defoghaze = dcp_defoghaze
    elif algo == 'improved_dcp':
        algo_parser_config = IMPROVED_DCP_PARSER_CONFIG
        defoghaze = idcp_defoghaze

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path of a foggy/hazy image(s) or a directory of ones to be processed')
    parser.add_argument('-gp', '--gt-path', help='Path of corresponding ground-truth image or a directory of corresponding ones used for assessment')
    parser.add_argument('-op', '--output-path', help='Path of a directory to store defoghazing results')
    parser.add_argument('-sm', '--save-mode', type=int, choices=(0, 1, 2, 3), default=0, help='0 - no results are saved, 1 - save only defoghazing results, 2 - save only performance report, 3 - save all results')
    add_algorithm_arguments(parser, algo_parser_config)

    kwargs = vars(parser.parse_args())
    print('kwargs:', kwargs)

    input_path = kwargs.pop('input_path')
    bgr_images = utils.read_images_from_path(input_path)
    print('Num of BGR images:', len(bgr_images))

    gt_path = kwargs.pop('gt_path')
    bgr_gts = None
    if gt_path:
        bgr_gts = utils.read_images_from_path(gt_path)
    
    output_path = kwargs.pop('output_path')
    save_mode = kwargs.pop('save_mode')

    # Remaining keys are used to pass to algorithm, and, None value will be removed
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    # Run algorithm with performance measurement
    results = {}
    psnrs = {}
    ssims = {}
    
    for i, img in bgr_images.items():
        print('==================================================')
        print(f'Proccess image "{i}" with shape {img.shape} ({img.shape[0] * img.shape[1]} pixels not considering all 3 channels)', '\n')

        start = time.perf_counter()
        dfh_result = defoghaze(img, **kwargs)
        end= time.perf_counter()
        dfh_result['recovered_bgr'] = utils.minmax_normalize(dfh_result['recovered_bgr'], (0, 1), (0, 255), np.uint8)
        results[i] = dfh_result
        
        elapsed_time = end-start
        print('Speed (1-time measurement):')
        print('-- Execution time (s):', elapsed_time)
        print('-- FPS: ', 1/(elapsed_time), '\n')

        if gt_path:
            psnr = sk_psnr(bgr_gts[i], dfh_result['recovered_bgr'])
            ssim = sk_ssim(cv.cvtColor(bgr_gts[i], cv.COLOR_BGR2GRAY), cv.cvtColor(dfh_result['recovered_bgr'], cv.COLOR_BGR2GRAY))
            print('Image Quality:')
            print('-- PSNR: ', psnr)
            print('-- SSIM: ', ssim)
            psnrs[i] = psnr
            ssims[i] = ssim
        print('==================================================')
    
    if gt_path:
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
            
        if gt_path and (save_mode == 2 or save_mode == 3):
            print(psnrs)
            print(avg_psnr)
            print(ssims)
            print(avg_ssim)
