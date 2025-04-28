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
from data_manip import load_data, train_file, test_file, one_hot_encode
from data_manip import load_points_from_stl, show_point_cloud_panda

# other imports
import keras
import os
import numpy as np

def load_all_data():
    """
    load training and test data from given files
    labels are classification labels while masks are point classification

    :return: training data, labels, masks and test data, labels, mask
    """
    train_data, train_label, train_mask = load_data(train_file)
    test_data, test_label, test_mask = load_data(test_file)

    return train_data, train_label, train_mask, test_data, test_label, test_mask

def build_pointnet_model(filepath=None):
    """
    Try to load a PointNet model from the given filepath. Otherwise build a model with the PointNet architecture

    :param filepath: path to model, defaults to None
    :return: model
    """
    if filepath is not None and os.path.exists(filepath):
        print(f'Loading model from {filepath}')
        model = keras.models.load_model(filepath)
    else:
        print(f'Model checkpoint does not exist')
        model = PointNetFull(num_points=2048, num_classes=2)
    print(type(model))
    model.summary()
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

    layers = [
        keras.Input(shape=input_size),
        keras.layers.Conv2D(64, kernel_size=(3,1), activation='relu', padding='same'),
        keras.layers.MaxPool2D(pool_size=(2,2), padding="same"),
        keras.layers.Conv2D(64, kernel_size=(3,1), activation='relu', padding='same'),
        keras.layers.MaxPool2D(pool_size=(2,1), padding="same"),

        keras.layers.UpSampling2D(size=(2,1)),
        keras.layers.Conv2DTranspose(64, kernel_size=(3,1), activation='relu', padding='same'),
        keras.layers.UpSampling2D(size=(2,1)),
        keras.layers.Conv2DTranspose(64, kernel_size=(3,1), activation='relu', padding='same'),

        keras.layers.Conv2D(output_size, kernel_size=(1,1), activation='softmax', padding='same')
    ]

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

def save_data(filename, model_name, scores, parameters, precision, recall):
    """
    save results of experiment to file

    :param filename: file to save to
    :param model_name: name of model run
    :param scores: training and test scores
    :param parameters: parameters used in experiment, list of strings
    """
    with open(filename, 'w') as f:
        f.write(f'Model,{model_name},\n')
        f.write(f'Parameters,\n')
        for i, param in enumerate(parameters):
            f.write(f'{param},')
            if (i-1)%2 == 0:
                f.write('\n')
        f.write(f'\nTrain Score,{scores[0]},Test score,{scores[1]},')
        f.write(f'Best Train Recall,{recall[0]},Best Test Recall,{recall[1]},')
        f.write(f'Best Train Precision,{precision[0]},Best Test Recall,{precision[1]},\n')

# ------ run experiments -------
train_data, train_label, train_mask, test_data, test_label, test_mask = load_all_data()
print(train_data.shape)
print(train_mask.shape)

# # -----------------
# # build and define fcnn model
# num_node_layers = [50 for i in range(0,20)]
# num_node_layers = [2000 - (i*75) for i in range(0,20)]
# model = build_fcnn_model(20, num_node_layers,'relu', (2048, 3), 2)
# model = Keras_Custom_Model(model, "fcnn")
# loss = None
# # --------------------

# # -----------------
# # build and define cnn model
# train_data, test_data = data_prep_cnn(train_data, test_data)    # expand dims for cnn
# train_onehot_labels = one_hot_encode(train_mask)                # one hot encode train labels
# test_onehot_labels = one_hot_encode(test_mask)                  # one hot encode test labels

# model = build_cnn_model(
#     2,
#     (2048, 3,1), output_size=2)
# model = Keras_Custom_Model(model, "cnn_onehot_customloss")
# loss = None
# # --------------------

# ----------
# build pointnet model
# should not do sparse categorical crossentropy
# onehot encode labels
model = build_pointnet_model()
model = Keras_Custom_Model(model, "pointnet_pr-re")
train_mask = np.eye(2)[train_mask]
loss = 'categorical_crossentropy'
# ----------



# ----- compile after architecture specified -------
model.build_callbacks()

# specify parameters to test
batch_sizes = [64] #, 32, 64]
epochs = [200] #, 50, 100]
optimizer = ['adam'] #, 'adamw']
validation_split = [.1] #, .2, .3]

# train model
for batch in batch_sizes:
    for epoch in epochs:
        for opt in optimizer:
            # compile model
            model.compile_model(opt, loss)
            for valid in validation_split:
                exp_name = f"{model.model_name}_b{batch}_e{epoch}_o{opt}_v{int(valid*100)}"
                params = ['batch', batch, 'epochs', epoch, 'optimizer', opt, 'validation', valid]
                print(f'\nNow running model: {exp_name}')

                # train model
                # history = model.train_model(train_data, train_onehot_labels, batch, epoch, valid)
                history = model.train_model(train_data, train_label, batch, epoch, valid)
                # score model
                # scores = model.score_model(train_data, train_onehot_labels, test_data, test_onehot_labels)
                scores = model.score_model(train_data, train_label, test_data, test_label)

                # calculate precision and recall
                best_train_precision = np.max(history.history['precision'])
                best_train_recall = np.max(history.history['recall'])
                best_val_precision = np.max(history.history['val_precision'])
                best_val_recall = np.max(history.history['val_recall'])

                # Print best epoch's values
                print(f"Best Training Precision: {best_train_precision:.4f}")
                print(f"Best Training Recall: {best_train_recall:.4f}")
                print(f"Best Validation Precision: {best_val_precision:.4f}")
                print(f"Best Validation Recall: {best_val_recall:.4f}")

                precision = [best_train_precision, best_val_precision]
                recall = [best_train_recall, best_val_recall]

                # save data
                save_data(f'exp5/{exp_name}.csv', exp_name, scores, params, precision, recall)

