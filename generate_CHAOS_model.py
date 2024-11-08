#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2021 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
import os
import sys
from dafne_dl import DynamicDLModel

if 'generate_convert' not in locals() and 'generate_convert' not in globals():
    from dafne_models.common import generate_convert

## CONFIGURATION. Automatically generated by the model trainer. Do not edit.
MODEL_NAME = "ChAoS"
MODEL_UID = "3cfdc49c-191a-467b-8c31-a0a6e056834b"
## END OF CONFIGURATION.

def make_unet():

    ## Configuration. Automatically generated by the model trainer. Do not edit.
    LABELS_DICT = {1: 'LK', 2: 'Liver', 3: 'RK', 4: 'Spleen'}
    MODEL_SIZE = [256, 256]
    N_LEVELS = 5
    N_CONV_LAYERS = 2
    INITIAL_KERNEL_SIZE = 2
    ## End of configuration.

    DEBUG = False

    def print_dbg(*args, **kwargs):
        if DEBUG:
            print(*args, **kwargs)

    print_dbg('N_LEVELS', N_LEVELS)

    OUTPUT_LAYERS = max(list(LABELS_DICT.keys())) + 1

    from tensorflow.keras import regularizers
    from tensorflow.keras.activations import softmax
    from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Concatenate, Lambda, Activation, Reshape, Add
    from tensorflow.keras.models import Model

    inputs=Input(shape=(MODEL_SIZE[0],MODEL_SIZE[1],2))
    weight_matrix=Lambda(lambda z: z[:,:,:,1])(inputs)
    weight_matrix=Reshape((MODEL_SIZE[0],MODEL_SIZE[1],1))(weight_matrix)
    reshape=Lambda(lambda z : z[:,:,:,0])(inputs)
    reshape=Reshape((MODEL_SIZE[0],MODEL_SIZE[1],1))(reshape)

    reg=0.01

    def make_down_layer(input_layer, filters, kernel_size, strides, n_conv_layers = 2):
        level = input_layer
        level = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_regularizer=regularizers.l2(reg))(input_layer)
        level = BatchNormalization(axis=-1)(level)
        level_shortcut = level  # Level1_l#
        level = Activation('relu')(level)
        cur_kernel_size = 3 + 2*(n_conv_layers-1)
        for i in range(n_conv_layers-1):
            level = Conv2D(filters=filters, kernel_size=(cur_kernel_size, cur_kernel_size), strides=1, padding='same',
                          kernel_regularizer=regularizers.l2(reg))(
                          level)  # (Level1_l)# ##  kernel_initializer='glorot_uniform' is the default
            cur_kernel_size -= 2
            level = BatchNormalization(axis=-1)(level)
            # Level1_l=InstanceNormalization(axis=-1)(Level1_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
            level = Activation('relu')(level)
        # Level1_l=Dropout(0.5)(Level1_l)
        level = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same',
                          kernel_regularizer=regularizers.l2(reg))(level)
        level = BatchNormalization(axis=-1)(level)
        # Level1_l=InstanceNormalization(axis=-1)(Level1_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
        level = Add()([level, level_shortcut])
        level = Activation('relu')(level)
        print_dbg("Down level with ", level.shape, ' filters ', filters, ' kernel_size ', kernel_size, ' strides ', strides)
        return level

    levels_down = []
    levels_down.append(make_down_layer(reshape, 32, (1, 1), 1, N_CONV_LAYERS))
    current_filter_size = 32
    current_kernel_size = int(INITIAL_KERNEL_SIZE)
    kernel_sizes = []
    for i in range(1, N_LEVELS):
        kernel_sizes.append(current_kernel_size)
        levels_down.append(make_down_layer(levels_down[-1], current_filter_size, current_kernel_size, 2, N_CONV_LAYERS))
        current_kernel_size -= 1
        if current_kernel_size < 2:
            current_kernel_size = 2
        current_filter_size *= 2

    print_dbg('Bottom level with ', current_filter_size, ' filters')
    bottom_level=Conv2D(filters=current_filter_size,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(levels_down[-1])
    bottom_level=BatchNormalization(axis=-1)(bottom_level)
    bottom_level=Activation('relu')(bottom_level)
    print_dbg('Bottom level size ', bottom_level.shape)
    kernel_sizes.append(2)

    def calc_padding(level_down, level_up, strides, kernel_size):
        output_size = level_up.shape[1] * strides + kernel_size - strides
        return level_down.shape[1] - output_size

    def make_up_layer(input_layer, down_layer, filters, kernel_size, strides, padding, n_conv_layers = 2):
        #print_dbg("Up level with input size ", input_layer.shape, 'down layer', down_layer.shape, 'filters ', filters, ' kernel_size ', kernel_size, ' strides ', strides, " output_padding ", padding)
        if padding <= 0:
            output_padding = None
        else:
            output_padding = (padding, padding)
        level_up = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, output_padding=output_padding,
                               kernel_regularizer=regularizers.l2(reg))(input_layer)
        level_up = BatchNormalization(axis=-1)(level_up)
        level_up_shortcut = level_up
        # Level4_r=InstanceNormalization(axis=-1)(Level4_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
        level_up = Activation('relu')(level_up)
        print_dbg("Up level with ", level_up.shape, ' filters ', filters, ' kernel_size ', kernel_size, ' strides ',
                  strides, "output_padding ", padding)
        level_up = Concatenate(axis=-1)([down_layer, level_up])
        cur_kernel_size = 3
        for i in range(n_conv_layers-1):
            level_up = Conv2D(filters=filters, kernel_size=(cur_kernel_size, cur_kernel_size), strides=1, padding='same',
                              kernel_regularizer=regularizers.l2(reg))(level_up)
            cur_kernel_size += 2
            level_up = BatchNormalization(axis=-1)(level_up)
            level_up = Activation('relu')(level_up)
        # Level4_r=Dropout(0.5)(Level4_r)
        level_up = Conv2D(filters=filters, kernel_size=(cur_kernel_size, cur_kernel_size), strides=1, padding='same',
                          kernel_regularizer=regularizers.l2(reg))(level_up)
        level_up = BatchNormalization(axis=-1)(level_up)
        # Level4_r=InstanceNormalization(axis=-1)(Level4_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
        level_up = Add()([level_up, level_up_shortcut])
        level_up = Activation('relu')(level_up)
        return level_up

    levels_up = []
    levels_up.append(bottom_level)
    for level in range(N_LEVELS - 1, -1, -1):
        current_filter_size //= 2
        down_level = levels_down[level]
        input_level = levels_up[-1]
        padding = calc_padding(down_level, input_level, 2, kernel_sizes[level])
        levels_up.append(make_up_layer(input_level, down_level, current_filter_size, kernel_sizes[level], 2, padding, N_CONV_LAYERS))

    output=Conv2D(filters=OUTPUT_LAYERS,kernel_size=(1,1),strides=1,kernel_regularizer=regularizers.l2(reg))(levels_up[-1])
    output=Lambda(lambda x : softmax(x,axis=-1))(output)
    output=Concatenate(axis=-1)([output,weight_matrix])
    model=Model(inputs=inputs,outputs=output)
    return model


def model_apply(modelObj: DynamicDLModel, data: dict):

    ## Configuration
    LABELS_DICT = {1: 'LK', 2: 'Liver', 3: 'RK', 4: 'Spleen'}
    MODEL_RESOLUTION = [2.04, 2.04]
    MODEL_SIZE = [256, 256]
    BIASCORRECTION_LEVELS = 4
    BIASCORRECTION_NORMALIZE = -1
    ## end of configuration

    from dafne_dl.common.padorcut import padorcut
    from dafne_dl.common import biascorrection

    import numpy as np
    from scipy.ndimage import zoom

    OUTPUT_LAYERS = max(list(LABELS_DICT.keys())) + 1
    MODEL_RESOLUTION = np.array(MODEL_RESOLUTION)

    netc = modelObj.model
    resolution = np.array(data['resolution'])
    zoomFactor = resolution/MODEL_RESOLUTION
    img = data['image']
    originalShape = img.shape
    img = zoom(img, zoomFactor) # resample the image to the model resolution

    img = padorcut(img, MODEL_SIZE)
    imgbc = biascorrection.biascorrection_image(img, BIASCORRECTION_LEVELS, BIASCORRECTION_NORMALIZE)

    segmentation = netc.predict(np.expand_dims(np.stack([imgbc, np.zeros(MODEL_SIZE)], axis=-1), axis=0))
    label = np.argmax(np.squeeze(segmentation[0, :, :, :OUTPUT_LAYERS]), axis=2)

    labelsMask = zoom(label, 1 / zoomFactor, order=0)
    labelsMask = padorcut(labelsMask, originalShape).astype(np.int8)

    outputLabels = {}

    for labelValue, labelName in LABELS_DICT.items():
        outputLabels[labelName] = (labelsMask == labelValue).astype(np.int8)  # left in the image is right in the anatomy

    return outputLabels


def model_incremental_mem(modelObj: DynamicDLModel, trainingData: dict, trainingOutputs,
                          bs=5, minTrainImages=5):

    ## Configuration
    LABELS_DICT = {1: 'LK', 2: 'Liver', 3: 'RK', 4: 'Spleen'}
    MODEL_RESOLUTION = [2.04, 2.04]
    MODEL_SIZE = [256, 256]
    BIASCORRECTION_LEVELS = 4
    BIASCORRECTION_NORMALIZE = -1
    ## end of configuration



    import dafne_dl.common.preprocess_train as pretrain
    from dafne_dl.common.DataGenerators import DataGeneratorMem
    from dafne_dl.labels.utils import invert_dict

    inverse_labels = invert_dict(LABELS_DICT)

    from tensorflow.keras import optimizers
    import time

    BAND = 49
    BATCH_SIZE = bs
    MIN_TRAINING_IMAGES = minTrainImages

    t = time.time()
    print('Image preprocess')

    image_list, mask_list = pretrain.common_input_process_single(inverse_labels, MODEL_RESOLUTION, MODEL_SIZE, MODEL_SIZE, trainingData,
                                                          trainingOutputs, False, BIASCORRECTION_LEVELS, BIASCORRECTION_NORMALIZE)

    print('Done. Elapsed', time.time()-t)
    nImages = len(image_list)

    if nImages < MIN_TRAINING_IMAGES:
        print("Not enough images for training")
        return

    print("image shape", image_list[0].shape)
    print("mask shape", mask_list[0].shape)

    print('Weight calculation')
    t = time.time()

    output_data_structure = pretrain.input_creation_mem(image_list, mask_list, BAND)

    print('Done. Elapsed', time.time() - t)

    card = len(image_list)
    steps = int(float(card) / BATCH_SIZE)

    print(f'Incremental learning with {nImages} images')
    t = time.time()

    netc = modelObj.model
    #checkpoint_files = os.path.join(CHECKPOINT_PATH, "weights - {epoch: 02d} - {loss: .2f}.hdf5")
    training_generator = DataGeneratorMem(output_data_structure, list_X=list(range(steps * BATCH_SIZE)), batch_size=BATCH_SIZE, dim=MODEL_SIZE)
    #check = ModelCheckpoint(filepath=checkpoint_files, monitor='loss', verbose=0, save_best_only=False,save_weights_only=True, mode='auto', period=10)
    #check = ModelCheckpoint(filepath=checkpoint_files, monitor='loss', verbose=0, save_best_only=True, # save_freq='epoch',
    #                        save_weights_only=True, mode='auto')
    adamlr = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
    netc.compile(loss=pretrain.weighted_loss, optimizer=adamlr)
    #history = netc.fit_generator(generator=training_generator, steps_per_epoch=steps, epochs=5, callbacks=[check], verbose=1)
    #history = netc.fit(x=training_generator, steps_per_epoch=steps, epochs=5, callbacks=[check],verbose=1)
    history = netc.fit(x=training_generator, steps_per_epoch=steps, epochs=5, verbose=1)
    print('Done. Elapsed', time.time() - t)


generate_convert(model_id=MODEL_UID,
                 default_weights_path=os.path.join('weights', f'weights_{MODEL_NAME}.weights.hd5'),
                 model_name_prefix=MODEL_NAME,
                 model_create_function=make_unet,
                 model_apply_function=model_apply,
                 model_learn_function=model_incremental_mem
                 )
