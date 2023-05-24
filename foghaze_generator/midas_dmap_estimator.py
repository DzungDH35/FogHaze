from base.depth_map_estimator import BaseDepthMapEstimator
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch

"""
_midas_model_type = 'DPT_Large'    # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
_midas_model_type = 'DPT_Hybrid'   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
_midas_model_type = 'MiDaS_small'  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
"""
VALID_MIDAS_MODEL_TYPE = ['DPT_Large', 'DPT_Hybrid', 'MiDaS_small']


"""
@class An implementation of depth map estimator which is, specifically, a Midas model.
"""
class MidasDmapEstimator(BaseDepthMapEstimator):
    _midas = None
    _transform = None
    _device = None


    # @param (mixture of np.ndarray and str) images
    # @param (str) midas_model_type
    def __init__(self, images, midas_model_type='DPT_Large'):
        super().__init__(images)

        if midas_model_type not in VALID_MIDAS_MODEL_TYPE:
            raise ValueError('Expect midas_model_type to be one of "DPT_Large" or "DPT_Hybrid" or "MiDaS_small"!')
        
        self._midas = torch.hub.load('intel-isl/MiDaS', midas_model_type)
        
        # Move model to GPU if available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._device = device
        self._midas.to(device)
        self._midas.eval()

        # Load transforms to resize and normalize the image for large or small model
        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        if midas_model_type == 'DPT_Large' or midas_model_type == 'DPT_Hybrid':
            self._transform = midas_transforms.dpt_transform
        else:
            self._transform = midas_transforms.small_transform


    # @return np.ndarray[]
    def estimate_depth_maps(self):
        depth_maps = []

        if len(self._base_images) == 0:
            print('No base images to estimate from!')
            return []
        
        # Preprocessing - Load image and apply transforms
        input_batches = []
        for img in self._base_images:
            input_batches.append(self._transform(img).to(self._device))
        
        # Predict and resize to original resolution
        predictions = []
        with torch.no_grad():
            for i, input_batch in enumerate(input_batches):
                prediction = self._midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=self._base_images[i].shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                predictions.append(prediction)

        for prediction in predictions:
            depth_maps.append(prediction.cpu().numpy())

        return depth_maps
    