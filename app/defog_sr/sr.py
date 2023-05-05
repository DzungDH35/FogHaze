import os.path

from PIL import Image
from PIL.Image import Resampling

from keras.preprocessing.image import img_to_array
from tensorflow import keras
from keras import layers
import tensorflow as tf

import cv2
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print('Not enough GPU hardware devices available, Run CPU mode')


class SuperResolution:
    def __init__(self, checkpoint_filepath=os.path.join(os.getcwd(), 'app/defog_sr/checkpoint/checkpoint')):
        self.checkpoint_filepath = checkpoint_filepath
        self.model = None
        self.load_super_resolution_model()

    @staticmethod
    def get_model(upscale_factor=3, channels=1):
        conv_args = {
            "activation": "relu",
            "kernel_initializer": "Orthogonal",
            "padding": "same",
        }
        inputs = keras.Input(shape=(None, None, channels))
        x = layers.Conv2D(64, 5, **conv_args)(inputs)
        x = layers.Conv2D(64, 3, **conv_args)(x)
        x = layers.Conv2D(32, 3, **conv_args)(x)
        x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
        outputs = tf.nn.depth_to_space(x, upscale_factor)

        return keras.Model(inputs, outputs)

    def load_super_resolution_model(self):
        self.model = self.get_model()
        self.model.load_weights(self.checkpoint_filepath)

    def upscale_image(self, img, denoise=False):
        channels = cv2.split(img)
        if len(channels) == 4:
            img = cv2.merge([channels[0], channels[1], channels[2]])

        img = Image.fromarray(img)
        """Predict the result based on input image and restore the image as RGB."""
        ycbcr = img.convert("YCbCr")
        y, cb, cr = ycbcr.split()
        y = img_to_array(y)
        y = y.astype("float32") / 255.0

        data = np.expand_dims(y, axis=0)
        out = self.model.predict(data)

        out_img_y = out[0]
        out_img_y *= 255.0

        # Restore the image in RGB color space.
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
        out_img_y = Image.fromarray(np.uint8(out_img_y), mode="L")
        out_img_cb = cb.resize(out_img_y.size, Resampling.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Resampling.BICUBIC)
        out_img = Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
            "RGB"
        )
        out_img = np.array(out_img)
        if denoise:
            out_img = cv2.fastNlMeansDenoisingColored(out_img, None, 3, 10, 7, 21)

        return out_img
