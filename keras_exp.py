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

def data_prep_cnn(train_imgs, test_imgs):
    """
    Change shape of images for convolutional network.

    :param train_imgs: np array of training images
    :param test_imgs: np array of test images
    :return: fixed arrays of training images and test images
    """
    cnn_train_imgs = np.expand_dims(train_imgs, -1)
    cnn_test_imgs = np.expand_dims(test_imgs, -1)

    return cnn_train_imgs, cnn_test_imgs

def build_cnn_model(num_conv_layers, input_size, output_size):
    """
    Assemble appropriate layers for a convolutional neural network.
    - Input layer
    - n amount of 'hidden' layers 
      - consists of a convolutional layer and a max pooling layer
    - Flatten, dropout, and softmax layer

    :param num_conv_layers: number of convolutional layers
    :param input_size: tuple of input shape
    :param output_size: number of categories
    :return: built model
    """
    layers = []

    # input layer
    layers.append(keras.Input(shape=input_size))
    # layers.append(keras.layers.Flatten(input_shape=input_size))

    # convolutional layer
    for i in range(0, num_conv_layers):
        layers.append(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
        layers.append(keras.layers.MaxPool2D(pool_size=(2,2), padding="same"))
    
    # transform to NN
    layers.append(keras.layers.Flatten())
    layers.append(keras.layers.Dropout(.5))
    layers.append(keras.layers.Dense(output_size, activation='softmax'))

    model = keras.models.Sequential(layers)
    model.summary()

    return model

def build_fcnn_model(num_layers:int, num_nodes:list, activation:str, input_size, output_size):
    """
    Build a fully connected neural network with the parameters given

    :param num_layers: number of hidden layers that should be created
    :param num_nodes: number of nodes each hidden layer should have
    :param activation: the activation function for each hidden layer
    :param input_size: what size the input layer should have
    :param output_size: what size the output layer should have
    :return: a model
    """
    # num nodes is a list of ints
    # list size must match num layers
    if len(num_nodes) != num_layers:
        print(f'node number list does not match: {len(num_nodes)} != {num_layers}')
        exit(1)
    
    # build model
    layers = []

    # input layer - providing shape of data (not batches)
    layers.append(keras.layers.Input(shape=input_size))

    # hidden layers
    for i in range(0, num_layers):
        layers.append(keras.layers.Dense(num_nodes[i], activation=activation))

    # output layer - same number of units as out output classes
    layers.append(keras.layers.Dense(output_size, activation="softmax"))

    # make model
    model = keras.models.Sequential(layers)

    # print model summary
    print(model.summary())
    
    return model

def load_keras_model(filename:str):
    if os.path.exists(filename):
        model = keras.models.load_model(filename)
    else:
        print(f"no model found at {filename}")
        model = None

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

# # ------ run experiments -------
# train_data, train_label, train_mask, test_data, test_label, test_mask = load_all_data()
# train_data, test_data = data_prep_cnn(train_data, test_data)
# # train_mask, test_mask = data_prep_cnn(train_mask, test_mask)
# print(train_data.shape)
# print(train_mask.shape)
# # model = build_pointnet_model("models/keras_checkpoint_keras_pointnet.keras")
# # model = build_dgcnn_model(train_data[0].shape)

# # num_node_layers = [50 for i in range(0,20)]
# # num_node_layers = [2000 - (i*75) for i in range(0,20)]

# model = build_cnn_model(
#     5,
#     (2048, 3, 1), output_size=2)
# # model = load_keras_model("model_checkpoints/keras_checkpoint_fully_connected_nn_custom_layers.keras")
# model = Keras_Custom_Model(model, "fully_connected_nn_custom_layers")
# model.build_callbacks()


# batch_sizes = [64] #, 32, 64]
# epochs = [200] #, 50, 100]
# optimizer = ['adam'] #, 'adamw']
# validation_split = [.1] #, .2, .3]

# for batch in batch_sizes:
#     for epoch in epochs:
#         for opt in optimizer:
#             # compile model
#             model.compile_model(opt)
#             for valid in validation_split:
#                 exp_name = f"{model.model_name}_b{batch}_e{epoch}_o{opt}_v{int(valid*100)}"
#                 params = ['batch', batch, 'epochs', epoch, 'optimizer', opt, 'validation', valid]
#                 print(f'\nNow running model: {exp_name}')

#                 # train model
#                 history = model.train_model(train_data, train_mask, batch, epoch, valid)
#                 # score model
#                 scores = model.score_model(train_data, train_mask, test_data, test_mask)

#                 # save data
#                 save_data(f'exp1/{exp_name}.csv', exp_name, scores, params)

# ----- prediction ---------
# predict on data and show point cloud
# model = PointNetFull("model_checkpoints/keras_checkpoint_pointnet_trainbatch2.keras", num_classes=2)
model = build_pointnet_model("model_checkpoints/keras_checkpoint_pointnet_trainbatch2.keras")
model = Keras_Custom_Model(model, "pointnet_trainingbatch2")
point_cloud = load_points_from_stl("data/knife.stl")

# perform random sampling of object
indices = np.array(range(len(point_cloud)))
indices = np.random.choice(indices, 2048, False)
model_subset = point_cloud[indices]
model_subset = np.expand_dims(model_subset, 0)
print(model_subset.shape)

pred = model.predict_model(model_subset)
print('pred', pred.shape, pred)

background_points = model_subset[pred[0]==1]
object_points = model_subset[pred[0]==0]
print('shapes:', background_points.shape, object_points.shape)

show_point_cloud_panda([background_points, object_points])
