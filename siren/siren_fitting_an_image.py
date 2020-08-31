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
"""SIREN for fitting an image [WIP]"""

import tensorflow as tf
print(tf.version.VERSION)

from PIL import Image
import numpy as np
import skimage
import time

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def get_meshgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range [-1,1]'''
    tensors = tuple(dim * [tf.linspace(-1, 1, num=sidelen)])
    meshgrid = tf.stack(tf.meshgrid(*tensors), axis=-1)
    meshgrid = tf.reshape(meshgrid, shape=[-1, dim])
    return meshgrid

class SineLayer(tf.keras.Model):
    """omega_0 is a frequency factor which multiplies the outputs prior activations.
    Since different signals may require different omega_0, it is a hyperparameter.
    """
    def __init__(self, in_features, num_features, bias=True, is_first=False, omega_0=30, **kwargs):
        super(SineLayer, self).__init__(**kwargs)
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features

        if self.is_first:
            initializer = tf.random_uniform_initializer(minval=-1 / self.in_features,
                                                        maxval=1 / self.in_features)
            self.dense = Dense(num_features, use_bias=bias, kernel_initializer=initializer)
        else:
            initializer = tf.random_uniform_initializer(
                minval=-tf.math.sqrt(6 / self.in_features) / self.omega_0,
                maxval=tf.math.sqrt(6 / self.in_features) / self.omega_0)
            self.dense = Dense(num_features, use_bias=bias, kernel_initializer=initializer)
        
    def call(self, inputs):
        return tf.math.sin(self.omega_0 * self.dense(inputs))

    def call_intermediate(self, inputs):
        intermediate = self.omega_0 * self.dense(inputs)
        return tf.math.sin(intermediate), intermediate

class Siren(tf.keras.Model):
    def __init__(self, in_features, hidden_features, hidden_layers, num_features, 
                 outer_dense=False, first_omega_0=30, hidden_omega_0=30., **kwargs):
        super(Siren, self).__init__(**kwargs)

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outer_dense:
            initializer = tf.random_uniform_initializer(
                minval=-tf.math.sqrt(6 / hidden_features) / hidden_omega_0,
                maxval=tf.math.sqrt(6 / hidden_features) / hidden_omega_0)
            final_dense = Dense(num_features)    

            self.net.append(final_dense)
        else:
            self.net.append(SineLayer(hidden_features, num_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = Sequential(*self.net)


    def forward(self, coords):
        coords = tf.Variable(coords.copy())
        output = self.net(coords)
        return output, coords

        def forward_with_activations(self, coords, retain_grad=False):
            '''Returns not only model output, but also intermediate activations.
            Only used for visualizing activations later'''
            activations = OrderedDict()

            activation_count = 0
            x = tf.Variable(coords.copy())
            activations['input'] = x
            for i, layer in enumerate(self.net):
                if isinstance(layer, SineLayer):
                    x, intermed = layer.forward_with_intermediate(x)
                        
                    activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                    activation_count += 1
                else: 
                    x = layer(x)
                        
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
                activation_count += 1

            return activations

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

@tf.function
def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += tf.gradeints(y[..., i], x, tf.ones_like(y[..., i]))[0][..., i:i+1]
    return div

@tf.function
def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = tf.ones_like(y)
    grad = tf.gradients(y, [x])[0]
    return grad

def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera()) 
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [sidelength, sidelength])
    img /= 255.
    return img

def build_train_tensors():
    img_mask_x = tf.random.uniform([sampled_pixel_count], maxval=rows, seed=0, dtype=tf.int32)
    img_mask_y = tf.random.uniform([sampled_pixel_count], maxval=cols, seed=1, dtype=tf.int32)

    img_mask_x = tf.expand_dims(img_mask_x, axis=-1)
    img_mask_y = tf.expand_dims(img_mask_y, axis=-1)

    img_mask_idx = tf.concat([img_mask_x, img_mask_y], axis=-1)
    img_train = tf.gather_nd(img_ground_truth, img_mask_idx, batch_dims=0)

    img_mask_x = tf.cast(img_mask_x, tf.float32) / rows
    img_mask_y = tf.cast(img_mask_y, tf.float32) / cols

    img_mask = tf.concat([img_mask_x, img_mask_y], axis=-1)

    return img_mask, img_train

if __name__ == '__main__':
    rows, cols, channels = 256, 256, 3
    sampled_pixel_count = int(256*256*0.1)

    img_mask, img_train = build_train_tensors()

    train_dataset = tf.data.Dataset.from_tensor_slices((img_mask, img_train))
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).cache()
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    cameraman = ImageFitting(256)
    img_siren = Siren(in_features=2, out_features=1, hidden_features=256, 
                    hidden_layers=3, outermost_linear=True)

    total_steps = 500 # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model_input, ground_truth = next(train_dataset))

    for step in range(total_steps):
        with tf.GradientTape() as tape:
            model_output, coords = img_siren(model_input)    
            loss = ((model_output - ground_truth)**2).mean()
        
            if not step % steps_til_summary:
                print("Step %d, Total loss %0.6f" % (step, loss))
                img_grad = gradient(model_output, coords)
                img_laplacian = laplace(model_output, coords)

                fig, axes = plt.subplots(1,3, figsize=(18,6))
                axes[0].imshow(tf.reshape(model_output.numpy(), [256,256]))
                axes[1].imshow(tf.reshape(img_grad.numpy().norm(dim=-1), [256,256]))
                axes[2].imshow(tf.reshape(img_laplacian.numpy(), 256,256))
                plt.show()

        grads = tape.gradient(loss, img_siren.trainable_parameters())
        optimizer.apply_gradients(zip(grads, img_siren.trainable_parameters()))

