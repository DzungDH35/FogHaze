from foghaze_removal.dcp import defoghaze as dcp_defoghaze
from foghaze_removal.msdcp import defoghaze as msdcp_defoghaze
from foghaze_removal.msialdcp import defoghaze as msialdcp_defoghaze
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from utilities.debug import plot_multiple_images
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import utilities.utilities as utils


DCP_PARSER_CONFIG = {
    'patch-size': {
        'short': 'ps',
        'type': int,
        'required': False,
        'help': 'Patch used to construct dark channel'
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

IADCP_PARSER_CONFIG = {
    **DCP_PARSER_CONFIG,
    'improved_al': {
        'short': 'ial',
        'type': int,
        'required': False,
        'choices': (0, 1),
        'help': 'Use improved estimation of atmospheric light (1) or not (0 - default). If not used, this algorithm is the original DCP!'
    },
    'al_resize_factor': {
        'short': 'arf',
        'type': float,
        'required': False,
        'help': 'Resize factor to estimate local atmospheric light faster (only affect when improved_al=True)'
    }
}
MSDCP_PARSER_CONFIG = {
    **DCP_PARSER_CONFIG,
    'fusion_weight': {
        'short': 'fw',
        'type': float,
        'required': False,
        'help': 'Fusion weight is used to blend transmission maps obtained from multi-scale analysis'
    }
}
MSIALDCP_PARSER_CONFIG = {
    **DCP_PARSER_CONFIG,
    'fusion_weight': {
        'short': 'fw',
        'type': float,
        'required': False,
        'help': 'Fusion weight is used to blend transmission maps obtained from multi-scale analysis'
    },
    'al_resize_factor': {
        'short': 'arf',
        'type': float,
        'required': False,
        'help': 'Resize factor to estimate local atmospheric light faster'
    }
}

SUPPORTED_ALGORITHMS = ('dcp', 'msdcp', 'msialdcp')

RELATIVE_DIR_DFH_RESULT = '/defoghazing_output/'
FILE_SUFFIX_DARK_CHANNEL = '_dc'
FILE_SUFFIX_BASE_TMAP = '_base_tmap'
FILE_SUFFIX_REFINED_TMAP = '_refined_tmap'
FILE_SUFFIX_RECOVERED = '_recovered'
FILE_NAME_PERF_REPORT = 'performance_report'

MAX_DISPLAYED_FIGURES = 3


def highlight_max(s):
    is_max = s == s.max()
    is_max['Average'] = False
    return ['color: #00ff00' if v else '' for v in is_max]


def highlight_min(s):
    is_min = s == s.min()
    is_min['Average'] = False
    return ['color: #ff0000' if v else '' for v in is_min]

def white_balance(image):
    image_float = image.astype(np.float32) / 255.0

    # Calculate the average color of the image
    avg_color = np.mean(image_float, axis=(0, 1))

    # Compute the scaling factors for each color channel
    gray_world = np.mean(avg_color)
    scaling_factors = gray_world / avg_color

    # Apply the scaling factors to each color channel
    balanced_image_float = image_float * scaling_factors
    
    balanced_image_float = np.clip(balanced_image_float, 0, 1)

    return (balanced_image_float * 255).astype(np.uint8)


def add_algorithm_arguments(parser: argparse.ArgumentParser, parser_config: dict):
    for arg_key, config in parser_config.items():
        parser.add_argument(
            f'-{config.pop("short", arg_key)}',
            f'--{arg_key}',
            **config
        )

def save_performance_report(df, output_path):
    performance_report_file = f"{os.path.join(output_path, FILE_NAME_PERF_REPORT)}.html"

    if not os.path.exists(performance_report_file):
        df.loc['Average'] = [
            np.mean(df['Speed(s) (one-time measurement)']), 
            np.mean(df['PSNR']), 
            np.mean(df['SSIM'])
        ]

        styled_df = df.style.apply(highlight_max, subset=['PSNR'], axis=0)\
                            .apply(highlight_min, subset=['PSNR'], axis=0)\
                            .apply(highlight_max, subset=['SSIM'], axis=0)\
                            .apply(highlight_min, subset=['SSIM'], axis=0)
        
        return styled_df.to_html(performance_report_file)
    
    old_df = pd.read_html(performance_report_file, index_col=0)[0].iloc[:-1]

    df = pd.concat([old_df, df])
    df.loc['Average'] = [
        np.mean(df['Speed(s) (one-time measurement)']), 
        np.mean(df['PSNR']), 
        np.mean(df['SSIM'])
    ]

    styled_df = df.style.apply(highlight_max, subset=['PSNR'], axis=0)\
                        .apply(highlight_min, subset=['PSNR'], axis=0)\
                        .apply(highlight_max, subset=['SSIM'], axis=0)\
                        .apply(highlight_min, subset=['SSIM'], axis=0)

    return styled_df.to_html(performance_report_file)

if __name__ == '__main__':
    print('Supported algorithms:', SUPPORTED_ALGORITHMS)
    algo = input('Select algorithm: ')
    print(algo)

    if algo not in SUPPORTED_ALGORITHMS:
        raise Exception('Not supported algorithm!')

    # Choose defoghazing algorithm for the main program
    if algo == 'dcp':
        algo_parser_config = IADCP_PARSER_CONFIG
        defoghaze = dcp_defoghaze
    elif algo == 'msdcp':
        algo_parser_config = MSDCP_PARSER_CONFIG
        defoghaze = msdcp_defoghaze
    elif algo == 'msialdcp':
        algo_parser_config = MSIALDCP_PARSER_CONFIG
        defoghaze = msialdcp_defoghaze

    # Build arguments for console
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path of a foggy/hazy image(s) or a directory of ones to be processed')
    parser.add_argument('-gp', '--gt-path', help='Path of corresponding ground-truth image or a directory of corresponding ones used for assessment')
    parser.add_argument('-op', '--output-path', help='Path of a directory to store defoghazing results')
    parser.add_argument('-sm', '--save-mode', type=int, choices=(0, 1, 2, 3), default=3, help='0 - no results are saved, 1 - save only defoghazing results, 2 - save only performance report, 3 - save all results (default)')
    parser.add_argument('-e', '--extra', type=int, choices=(0, 1), default=0, help='0 - only restored image (default), 1 - more defoghaze-related results')
    parser.add_argument('-dm', '--display-mode', type=int, choices=(0, 1), default=1, help='0 - no results are displayed, 1 - display defoghazing results within limit (maximum figures) (default)')
    parser.add_argument('-pp', '--post-processing', type=int, choices=(0, 1), default=1, help='Post processing or not (default is 1)')
    add_algorithm_arguments(parser, algo_parser_config)

    kwargs = vars(parser.parse_args())
    print('kwargs:', kwargs)

    input_path = kwargs.pop('input_path')
    bgr_images = utils.read_images_from_path(input_path)

    gt_path = kwargs.pop('gt_path')
    bgr_gts = {}
    if gt_path:
        bgr_gts = utils.read_images_from_path(gt_path)

    output_path = kwargs.pop('output_path')
    save_mode = kwargs.pop('save_mode')
    display_mode = kwargs.pop('display_mode')
    extra = kwargs.pop('extra')
    post_processing = kwargs.pop('post_processing')

    # Remaining keys are used to pass to algorithm, and, None value will be removed
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    # Run algorithm with performance measurement
    results = {}
    runtime = {}
    psnrs = {}
    ssims = {}
    
    for i, img in bgr_images.items():
        print('==================================================')
        print(f'Proccess image "{i}" with shape {img.shape} ({img.shape[0] * img.shape[1]} pixels not considering all 3 channels)', '\n')

        start = time.perf_counter()
        dfh_result = defoghaze(img, **kwargs)
        end= time.perf_counter()
        dfh_result['recovered_bgr'] = utils.minmax_normalize(dfh_result['recovered_bgr'], (0, 1), (0, 255), np.uint8)
        if post_processing:
            dfh_result['recovered_bgr'] = white_balance(dfh_result['recovered_bgr'])
            dfh_result['recovered_bgr'] = cv.fastNlMeansDenoisingColored(dfh_result['recovered_bgr'], hColor=20, h=5)
        results[i] = dfh_result
        
        elapsed_time = end-start
        print('Speed (1-time measurement):')
        print('-- Execution time (s):', elapsed_time)
        print('-- FPS: ', 1/(elapsed_time), '\n')
        runtime[i] = elapsed_time

        if i in bgr_gts:
            psnr = sk_psnr(bgr_gts[i], dfh_result['recovered_bgr'])
            ssim = sk_ssim(cv.cvtColor(bgr_gts[i], cv.COLOR_BGR2GRAY), cv.cvtColor(dfh_result['recovered_bgr'], cv.COLOR_BGR2GRAY))
            print('Image Quality:')
            print('-- PSNR: ', psnr)
            print('-- SSIM: ', ssim)
            psnrs[i] = psnr
            ssims[i] = ssim
        print('==================================================')
    
    if gt_path:
        avg_psnr = None if len(psnrs) == 0 else np.mean(list(psnrs.values()))
        avg_ssim = None if len(ssims) == 0 else np.mean(list(ssims.values()))
        print('Average PSNR:', avg_psnr)
        print('Average SSIM:', avg_ssim)
    
    if save_mode:
        output_path = output_path or os.path.dirname(input_path) + RELATIVE_DIR_DFH_RESULT
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        
        if save_mode == 1 or save_mode == 3:
            for i, result in results.items():
                fname, ext = os.path.splitext(i)
                fname_recovered = fname + FILE_SUFFIX_RECOVERED + '.jpg'

                cv.imwrite(os.path.join(output_path, fname_recovered), result['recovered_bgr'])

                if extra:
                    fname_dc = fname + FILE_SUFFIX_DARK_CHANNEL + ext
                    fname_base_tmap = fname + FILE_SUFFIX_BASE_TMAP + ext
                    fname_refined_tmap = fname + FILE_SUFFIX_REFINED_TMAP + ext
                    plt.imsave(os.path.join(output_path, fname_dc), result['dark_channel'], cmap='gray')
                    plt.imsave(os.path.join(output_path, fname_base_tmap), result['base_tmap'], cmap='gray')
                    plt.imsave(os.path.join(output_path, fname_refined_tmap), result['refined_tmap'], cmap='gray')

        if gt_path and (save_mode == 2 or save_mode == 3):
            df = pd.DataFrame({'Speed(s) (one-time measurement)': runtime, 'PSNR': psnrs, 'SSIM': ssims})
            save_performance_report(df, output_path)

    # Plot results
    if display_mode == 1:
        cnt = 0
        if len(results) > MAX_DISPLAYED_FIGURES:
            print(f'Display only {MAX_DISPLAYED_FIGURES}/{len(results)} defoghazing results!')
        for i, result in results.items():
            if cnt == MAX_DISPLAYED_FIGURES:
                break
            plot_multiple_images([
                cv.cvtColor(bgr_images[i], cv.COLOR_BGR2RGB),
                cv.cvtColor(result['recovered_bgr'], cv.COLOR_BGR2RGB),
                result['base_tmap'],
                result['refined_tmap']
            ])
            cnt += 1
