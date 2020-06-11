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
"""Unsupervised Hierarchical Disentanglement for Fine Grained Object Generation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

assert tf.version.VERSION.startswith('2.2')

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, ReLU, Activation
from tensorflow.keras.layers import UpSampling2D, Conv2D, Concatenate, Dense, concatenate
from tensorflow.keras.layers import Flatten, Lambda, Reshape, ZeroPadding2D, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class GLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GLU, self).__init__(**kwargs)

    def call(self, x):
        num_channels = x.shape[-1]
        assert num_channels % 2 == 0, "Channels don't divide by 2"
        num_channels /= 2
        return x[:, :, :, :num_channels] * Activation('sigmoid')(x[:, :, :, num_channels:])


def child_to_parent(child_code, child_classes, parent_classes):
    """Returns the parent conditional code"""
    ratio = child_classes/parent_classes
    arg_parent = tf.math.argmax(child_code, axis=1)/ratio
    parent_code = tf.zeros([child_code.shape[0], parent_classes])
    for i in range(child_code.shape[0]):
        parent_code[i][arg_parent[i]] = 1
    return parent_code

def conv3x3(filters=16):
    return Conv2D(filters=filters, kernel_size=3, strides=1, kernel_initializer="he_normal", 
            use_bias=False)


class UpSampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters=16, **kwargs):
        # Decide on the args for this block 
        super(UpSampleBlock, self).__init__(**kwargs)
        self.filters = filters
        
        @tf.function
        def call(self, inputs):
            x = UpSampling2D(size=2, interpolation="nearest")(inputs)
            x = conv3x3(self.filters * 2)(x)
            x = BatchNormalization()(x)
            return  GLU()(x)

class KeepDimsBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(KeepDimsBlock, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        pass
