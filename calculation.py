# ------------------------
# @file     calcuation.py
# @date     April, 2025
# @author   Jordan Reed
# @email    reed5204@vandals.uidaho.edu
# @brief    calculate precision and recall for models
#           
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


train_data, train_label, train_mask, test_data, test_label, test_mask = load_all_data()

# # calc metrics for pointnet
# model = Keras_Custom_Model(None, "pointnet")
# model.load_keras_model("./models/keras_checkpoint_keras_pointnet.keras")

# # calc metrics for fcnn1
# model = Keras_Custom_Model(None, "fcnn1")
# model.load_keras_model("./models/keras_checkpoint_fully_connected_nn.keras")

# calc metrics for fcnn2
model = Keras_Custom_Model(None, "fcnn2")
model.load_keras_model("./models/keras_checkpoint_fully_connected_nn_custom_layers.keras")

# # calc metrics for cnn
# model = Keras_Custom_Model(None, "cnn")
# model.load_keras_model("./models/keras_checkpoint_cnn_onehot.keras")
# test_mask = one_hot_encode(test_mask)


# print(model.model.summary())

model.predict_set(test_data[0:1], test_mask[0:1])