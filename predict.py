# ------------------------
# @file     data_manip.py
# @date     November, 2024
# @author   Jordan Reed
# @email    reed5204@vandals.uidaho.edu
# @brief    predict using saved models
# ------------------------

from keras_model_class import Keras_Custom_Model
from data_manip import load_data, test_file, class_dict
from data_manip import show_point_cloud_panda, show_point_clouds, load_points_from_stl
import numpy as np
from keras_model_class import Keras_Custom_Model

import matplotlib.pyplot as plt


# -------------
# load model
model = Keras_Custom_Model(model=None, model_name="fcnn_custom")
model.load_keras_model("models/keras_checkpoint_fully_connected_nn_batch2.keras")
model.model.summary()
# ------------

# ------ predict out of test set ---------
# load data
test_data, test_labels, test_mask = load_data(test_file)

index = [5, 200, 555, 999]
for i in index:
    object_model = test_data[i]
    object_bg_mask = test_mask[i]
    object_label = class_dict[f'{test_labels[i]}']

    print(f'Object is part of a {object_label}.')

    # run through model
    object_model = np.expand_dims(object_model, 0)      # give it a number of objects of 1
    prediction = model.predict_model(object_model)      # perform prediction
    # print(f'pred shape: {prediction.shape}')
    # print(prediction[0])
    # print((prediction==1).sum())
    # print(np.unique(prediction))

    print(f'Number background objects predicted: {(prediction==1).sum()}')
    object_model = np.reshape(object_model, (2048, 3))  # get rid of number of objects

    acc = (object_bg_mask==prediction).sum()            # count accuracy of prediction
    acc = acc / object_bg_mask.shape[0]
    print(f"accuracy for object: {acc:.2f}")

    # print(object_model.shape)

    # break into background and foreground
    background_object = object_model[prediction==1]
    foreground_object = object_model[prediction!=1]

    # ground truth
    truth_bg = object_model[object_bg_mask==1]
    truth_fg = object_model[object_bg_mask==0]
    # print(truth_bg.shape, truth_fg.shape)

    actual_background_points = (object_bg_mask==1).sum()
    pretend_accuracy = (2048-actual_background_points)/2048
    print(f'accuracy if every point is foreground: {pretend_accuracy:.2f}')
    # print(f'background objects: {(object_bg_mask==1).sum()}')
    # print(f'unique: {np.unique(object_bg_mask)}')

    background_object = np.resize(background_object, (2048, 3))
    foreground_object = np.resize(foreground_object, (2048, 3))
    truth_bg = np.resize(truth_bg, (2048, 3))
    truth_fg = np.resize(truth_fg, (2048, 3))

    # print('predicted shapes', background_object.shape, foreground_object.shape)
    # print('ground truth shapes', truth_bg.shape, truth_fg.shape)

    # show ground truth
    # show_point_clouds([truth_bg, truth_fg])

    # show prediction
    show_point_clouds([background_object, foreground_object])
# -------------

# # -------------
# # stl prediction
# # point_cloud = load_points_from_stl("data/eot_metal.stl", True)
# point_cloud = load_points_from_stl("data/tape.stl", True) 

# # perform random sampling of object
# indices = np.array(range(len(point_cloud)))
# indices = np.random.choice(indices, 2048, False)
# object_model = point_cloud[indices]
# print(f"from {point_cloud.shape} to {object_model.shape}")

# # fix shape for prediction
# object_model = np.expand_dims(object_model, 0)      # give it a number of objects of 1

# # get predictions
# prediction = model.predict_model(object_model)      # perform prediction

# print(f'Number background objects predicted: {(prediction==1).sum()}')
# object_model = np.reshape(object_model, (2048, 3))  # get rid of number of objects

# # break into background and foreground
# background_object = object_model[prediction==1]
# foreground_object = object_model[prediction!=1]

# # show prediction
# show_point_clouds([background_object, foreground_object])
# # -----------------
