# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Dataset functions for the FineGAN Model.
"""

import absl
import PIL
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from .config.config import Config

def get_images(path, size, bbox=None, normalize=None):

    img = Image.open(path).convert('RGB')
    width, height = img.size

    if bbox:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        fimg = img.copy()
        fimg_arr = np.array(fimg)
        fimg = Image.fromarray(fimg_arr)
        cimg = img.crop([x1, y1, x2, y2])

    retf = []
    retc = []
    re_cimg = tf.image.resize(cimg, size)
    retc.append(re_cimg)

    # TODO: Random Crop + Flip and Modify bbox accordingly
    
    # re_fimg = tf.image.resize(fimg, size=(126*76/64))
    # re_width, re_height = re_fimg.size

    retf.append(normalize(img))

    return retc, retf, bbox


class Dataset():
    def __init__(self, cfg, data_dir, base_size, **kwargs):

        # TODO: Apply normalizations and transforms

        self.imsize = []
        for _ in range(cfg.TREE['BRANCH_NUM']):
            self.imsize.append(base_size)
            base_size *= 2

        self.data = []
        self.data_die = data_dir
        self.bbox = self.load_bbox()
        self.filenames = self.load_filenames()

    def load_bbox(self):
        return None

    def load_filenames(self):
        return None
