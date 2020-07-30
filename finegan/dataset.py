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

import os
import absl
import PIL
import numpy as np
import pandas as pd
import tensorflow as tf

from .config.config import Config


def get_images(path, size, bbox=None, normalize=None):
    
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    width, height = tf.shape(img).numpy()[0], tf.shape(img).numpy()[1]

    if bbox:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        boxes = tf.constant([y1,x1,y2,x2], dtype=tf.float32)
        size = tf.constant([y2-y1, x2-x1], dtype=tf.int32)
        fimg = img
        # fimg_arr = np.array(fimg)
        # fimg = Image.fromarray(fimg_arr)
        cimg = tf.image.crop_and_resize(tf.reshape(img, [1, width, height, 3]), tf.reshape(boxes, [1, 4]), box_indices=[0], crop_size=size)

    retf = []
    retc = []
    size = []
    re_cimg = tf.image.resize(cimg, size)
    retc.append(re_cimg)

    # TODO: Random Crop + Flip and Modify bbox accordingly

    # re_fimg = tf.image.resize(fimg, size=(126*76/64))
    # re_width, re_height = re_fimg.size

    retf.append(tf.keras.utils.normalize(fimg, axis=-1, order=1))

    return retc, retf, bbox


class Dataset():
    def __init__(self, cfg, data_dir='..\..\CUB data\CUB_200_2011', base_size=64, **kwargs):

        # TODO: Apply normalization and transforms
        self.imsize = []
        for _ in range(cfg.TREE['BRANCH_NUM']):
            self.imsize.append(base_size)
            base_size *= 2

        self.data = []
        self.data_dir = data_dir
        self.iterator = self.train_pairs
        self.bbox = self.load_bbox()
        self.filenames = self.load_filenames()

    def load_bbox(self):
        bbox_path = os.path.join(self.data_dir, 'bounding_boxes.txt')
        bbox_df = pd.read_csv(bbox_path, delim_whitespace=True, header=None).astype(int)
        images_path = os.path.join(self.data_dir, 'images.txt')

        filenames_df = pd.read_csv(images_path, delim_whitespace=True, header=None)
        images_filenames = filenames_df[1].tolist()
        print(f'[INFO] Total Images: {len(images_filenames)} {images_filenames[0]}')

        bbox_dict = {image[:-4]: [] for image in images_filenames}
        num_images = len(images_filenames)
        for i in range(num_images):
            bbox = bbox_df.iloc[i][1:].tolist()
            key = images_filenames[i][:-4]
            bbox_dict[key] = bbox

        return bbox_dict

    def load_filenames(self):
        images_path = os.path.join(self.data_dir, 'images.txt')
        filenames_df = pd.read_csv(images_path, delim_whitespace=True, header=None)
        images_filenames = filenames_df[1].tolist()
        images_filenames = [image[:-4] for image in images_filenames]
        print(f'[INFO] Load filenames from: {images_path} {len(images_filenames)}')
        return images_filenames

    def train_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None

        image_name = f'{self.data_dir}/images/{key}.jpg'
        fimgs, cimgs, mod_bbox = get_images(image_name, self.imsize, bbox)

        rand_class= list(np.random.choice(range(200), 1))
        child_code = np.zeros([200,])
        child_code[rand_class] = 1

        return fimgs, cimgs, child_code, key, mod_bbox

    def get_item(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)

