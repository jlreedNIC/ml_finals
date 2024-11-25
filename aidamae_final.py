#==============================================================================
# Imported Modules
#==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # The GPU id to use, usually either "0" or "1"

import numpy as np

import keras
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Reshape
from keras.layers import Convolution1D, BatchNormalization, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Lambda, concatenate
import tensorflow as tf

#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================
def exp_dim(global_feature, num_points):
    return tf.tile(global_feature, [1, num_points, 1])

def extend_dimension(global_feature, axis):
    '''
    Extend dimension of a tensor(example: [None, 1024] to [None, 1, 1024])
    :param global_feature:
    :param axis:
    :return:
    '''
    return tf.expand_dims(global_feature, axis)

def extend_size(global_feature, num_points):
    '''
    Extend size of a tensor(example: [None, 1, 1024] to [None, num_points, 1024])
    :param global_feature:
    :param num_points:
    :return:
    '''
    return tf.tile(global_feature, [1, num_points, 1])

def multilayer_perceptron(inputs, mlp_nodes):
    '''
    Define multilayer-perceptron
    :param inputs: a tensor of input data
    :param layer_nodes: an array of intergers that defines num-nodes for each layer(example: [16, 16, 32, 32, 64], ...)
    :return: outputs of each layer
    '''
    mlp = []
    x = inputs
    for i, num_nodes in enumerate(mlp_nodes):
        x = Convolution1D(filters=num_nodes, kernel_size=1, activation='relu')(x)
        x = BatchNormalization()(x)
        mlp.append(x)
    return mlp


def TNet(inputs, tsize, mlp_nodes=(64, 128, 1024), fc_nodes=(512, 256)):
    '''
    Define T-Net(joint aligment network) to predict affine transformation matrix
    :param inputs: a tensor of input data
    :param tsize: an integer that defines the size of transformation matrix
    :param mlp_nodes: an array of intergers that defines num-nodes for each layer(example: [16, 16, 32, 32, 64], ...)
    :param fc_nodes:an array of intergers that defines num-nodes for each layer(example: [16, 16, 32, 32, 64], ...)
    :return:
    '''
    x = inputs
    for i, num_nodes in enumerate(mlp_nodes):
        x = Convolution1D(filters=num_nodes, kernel_size=1, activation='relu')(x)
        x = BatchNormalization()(x)

    x = GlobalMaxPooling1D()(x)

    for i, num_nodes in enumerate(fc_nodes):
        x = Dense(num_nodes, activation='relu')(x)
        x = BatchNormalization()(x)

    x = Dense(tsize*tsize, weights=[np.zeros([num_nodes, tsize*tsize]), np.eye(tsize).flatten().astype(np.float32)])(x) # constrain the feature transformation matrix to be close to orthogonal matrix
    transformation_matrix = Reshape((tsize, tsize))(x)
    return transformation_matrix


def PointNetFull(num_points, num_classes, type='seg'):
    '''
    Pointnet full architecture
    :param num_points: an integer that is the number of input points
    :param num_classes: an integer that is number of categories
    :param type: a string of 'seg' or 'cls' to select Segmentation network or Classification network
    :return:
    '''

    inputs = Input(shape=(num_points, 3))

    '''
    Begin defining Pointnet Architecture
    '''
    tnet1 = TNet(inputs=inputs, tsize=3, mlp_nodes=(128, 128, 1024), fc_nodes=(512, 256))
    aligned_feature1 = keras.layers.dot(inputs=[inputs, tnet1], axes=2)

    extracted_feature11, extracted_feature12, extracted_feature13 = multilayer_perceptron(inputs=aligned_feature1,
                                                                                          mlp_nodes=(64, 128, 128))

    tnet2 = TNet(inputs=inputs, tsize=128, mlp_nodes=(128, 128, 1024), fc_nodes=(512, 256))
    aligned_feature2 = keras.layers.dot(inputs=[extracted_feature13, tnet2], axes=2)

    extracted_feature21, extracted_feature22 = multilayer_perceptron(inputs=aligned_feature2, mlp_nodes=(512, 2048))

    global_feature = GlobalMaxPooling1D()(extracted_feature22)

    global_feature_seg = Lambda(extend_dimension, arguments={'axis': 1})(global_feature)
    global_feature_seg = Lambda(extend_size, arguments={'num_points': num_points})(global_feature_seg)

    # Classification block
    cls = Dense(512, activation='relu')(global_feature)
    cls = BatchNormalization()(cls)
    cls = Dense(256, activation='relu')(cls)
    cls = BatchNormalization()(cls)
    cls = Dense(num_classes, activation='softmax')(cls)
    cls = BatchNormalization()(cls)

    # Segmentation block
    seg = concatenate([extracted_feature11, extracted_feature12, extracted_feature13, aligned_feature2, extracted_feature21, global_feature_seg])
    _, _, seg  = multilayer_perceptron(inputs=seg, mlp_nodes=(256, 256, 128))
    seg = Convolution1D(num_classes, 1, padding='same', activation='softmax')(seg)
    '''
    End defining Pointnet Architecture
    '''

    if type=='seg':
        model = Model(inputs=inputs, outputs=seg)
    elif type=='cls':
        model = Model(inputs=inputs, outputs=cls)
    else:
        raise ValueError("ERROR!!! 'type' must be 'seg' or 'cls'")
    print(model.summary())

    return model

# import keras
import tensorflow as tf
import os
import h5py as hp
import numpy as np
import keras
from keras.layers import Input
from keras.models import Model

# folder is located:
test_file = "data/test_objectdataset_augmented25rot.h5"

train_file = "data/training_objectdataset_augmented25rot.h5"

def load_data(filename):
    print(f"Starting load data for {filename}")
    with hp.File(filename) as f:
        data = np.array(f['data'][:])
        labels = np.array(f['label'][:])
        mask = np.array(f['mask'][:])

    mask[mask==-1] = 15
    print(np.unique(mask))
    return data, labels, mask

import os
def build_model(filepath=None):
    if filepath is not None and os.path.exists(filepath):
        print(f'Loading model from {filepath}')
        model = keras.models.load_model(filepath)
    else:
        print(f'Model checkpoint does not exist')
        model = PointNetFull(num_points=2048, num_classes=16)
    print(type(model))
    model.summary()
    return model

def build_callbacks(model_type="model"):
    model_checkpoint_string = f'model_checkpoints/{model_type}.keras'

    print(model_checkpoint_string)

    callback_function = keras.callbacks.EarlyStopping(patience=2, monitor='loss')
    callback_function_2 = keras.callbacks.ModelCheckpoint(
        model_checkpoint_string,
        monitor='loss',
        mode='min',
        save_best_only=True
    )

    return [callback_function, callback_function_2]

def compile_model(model, optimizer='adam'):
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, train_data, train_labels, batch_size=16, epochs=10, callback_funcs=[], validation_split=0):
    history = model.fit(
        train_data, train_labels,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callback_funcs,
        validation_split=validation_split
        )

    return model, history

def score_model(model, train_data, train_labels, test_data, test_labels):
    train_score = model.evaluate(x=train_data, y=train_labels, verbose=0)
    test_score = model.evaluate(x=test_data, y=test_labels, verbose=0)

    print(f"training score: {train_score}")
    print(f'test score: {test_score}')

    return train_score, test_score

def save_data(filename, model_name, scores, parameters):
    with open(filename, 'w') as f:
        f.write(f'Model,{model_name},\n')
        f.write(f'Parameters,\n')
        for i, param in enumerate(parameters):
            f.write(f'{param},')
            if (i-1)%2 == 0:
                f.write('\n')
        f.write(f'\nTrain Score,{scores[0]},Test score,{scores[1]},\n')

# load data
train_data, train_label, train_mask = load_data(train_file)
test_data, test_label, test_mask = load_data(test_file)

# build model and callbacks
# model = build_model()
# gpu_model_checkpoint = "/content/drive/MyDrive/school/colab_things/model_checkpoints/pointnet.keras"
model_checkpoint_string = "model_checkpoints/pointnet.keras"
model = build_model(model_checkpoint_string)

callbacks = build_callbacks("pointnet")
batch_sizes = [128] # [16, 64, 128]
epochs = [200] #[10, 50, 100]
optimizer = ['adam'] # , 'adamw']
validation_split = [.1, .2, .3]

for batch in batch_sizes:
    for epoch in epochs:
        for opt in optimizer:
            # compile model
            model = compile_model(model, opt)

            for valid in validation_split:
                model_name = f"pointnet_b{batch}_e{epoch}_o{opt}_v{int(valid*100)}"
                params = ['batch', batch, 'epochs', epoch, 'optimizer', opt, 'validation', valid]
                print(f'\nNow running model: {model_name}')

                # train model
                model, history = train_model(model, train_data, train_mask, batch, epoch, callbacks, valid)
                # score model
                scores = score_model(model, train_data, train_mask, test_data, test_mask)
                # scores = [0,0]

                # save data
                # save_data(f'/content/drive/MyDrive/school/colab_things/{model_name}.csv', model_name, scores, params)
                save_data(f'results/{model_name}.csv', model_name, scores, params)