# ------------------------
# @file     keras_exp.py
# @date     November 20, 2024
# @author   Jordan Reed
# @email    reed5204@vandals.uidaho.edu
# @brief    model manipulation for final project
#           model built by NGUYEN CONG MINH
#           https://github.com/minhncedutw/pointnet1_keras
# ------------------------

# import models
from  other_libraries_used.pointnet import PointNetFull
# from other_libraries_used.pointconvTF2.model_modelnet import PointConvModel

# import model class
from keras_model_class import Keras_Custom_Model

# import data manipulation functions
from data_manip import load_data, train_file, test_file
from data_manip import load_points_from_stl, show_point_cloud_panda

# other imports
import keras
import os
from dgcnn.components import DeepGraphConvolution
import numpy as np

def load_all_data():
    train_data, train_label, train_mask = load_data(train_file)
    test_data, test_label, test_mask = load_data(test_file)

    return train_data, train_label, train_mask, test_data, test_label, test_mask

def build_pointnet_model(filepath=None):
    if filepath is not None and os.path.exists(filepath):
        print(f'Loading model from {filepath}')
        model = keras.models.load_model(filepath)
    else:
        print(f'Model checkpoint does not exist')
        model = PointNetFull(num_points=2048, num_classes=2)
    print(type(model))
    model.summary()
    return model

def build_pointconv_model(batch_size, input_shape):
    model = PointConvModel(batch_size, num_classes=2)
    model.build(batch_size, input_shape)

    return model

def save_data(filename, model_name, scores, parameters):
    with open(filename, 'w') as f:
        f.write(f'Model,{model_name},\n')
        f.write(f'Parameters,\n')
        for i, param in enumerate(parameters):
            f.write(f'{param},')
            if (i-1)%2 == 0:
                f.write('\n')
        f.write(f'\nTrain Score,{scores[0]},Test score,{scores[1]},\n')

def build_dgcnn_model(input_shape=None, filepath=None):
    if filepath is not None and os.path.exists(filepath):
        print(f'Loading dgcnn model from {filepath}')
        model = keras.models.load_model(filepath)
    elif input_shape is None:
        print(f'Input shape must not be none if filepath is none.')
        exit(1)
    else:
        print(f'Model checkpoint does not exists. Building model with shape {input_shape}')

        # corresponding fully connected adjacency matrices
        adjacency = np.ones((100, 10, 10))

        # inputs to the DGCNN
        X = keras.layers.Input(input_shape, name="graph_signal")
        E = keras.layers.Input(shape=(input_shape[0], input_shape[0]), name="adjacency")

        # DGCNN
        # Note that we pass the signals and adjacencies as a tuple.
        # The graph signal always goes first!
        output = DeepGraphConvolution([input_shape[0], 2], k=5 )((X, E))
        model = keras.Model(inputs=[X, E], outputs=output)

    return model

# ------ run experiments -------
train_data, train_label, train_mask, test_data, test_label, test_mask = load_all_data()

# model = build_pointnet_model("models/keras_checkpoint_keras_pointnet.keras")
model = build_dgcnn_model(train_data[0].shape)
# model = Keras_Custom_Model(model, "pointnet")
model = Keras_Custom_Model(model, "dgcnn")
model.build_callbacks()


batch_sizes = [16] #, 32, 64]
epochs = [10] #, 50, 100]
optimizer = ['adam'] #, 'adamw']
validation_split = [.1] #, .2, .3]

for batch in batch_sizes:
    for epoch in epochs:
        for opt in optimizer:
            # compile model
            model.compile_model(opt)
            for valid in validation_split:
                model_name = f"dgcnn_b{batch}_e{epoch}_o{opt}_v{int(valid*100)}"
                params = ['batch', batch, 'epochs', epoch, 'optimizer', opt, 'validation', valid]
                print(f'\nNow running model: {model_name}')

                # train model
                history = model.train_model(train_data, train_mask, batch, epoch, valid)
                # score model
                scores = model.score_model(train_data, train_mask, test_data, test_mask)

                # save data
                save_data(f'exp1/{model_name}.csv', model_name, scores, params)

# ----- prediction ---------
# predict on data and show point cloud
# point_cloud = load_points_from_stl("data/knife.stl")
# pred = model.predict_model(point_cloud)
# print('pred', pred.shape, pred)

# background_points = point_cloud[pred==1]
# object_points = point_cloud[pred==0]
# print('shapes:', background_points.shape, object_points.shape)

# show_point_cloud_panda([background_points, object_points])
