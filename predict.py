# from keras_model_class import Keras_Custom_Model
from data_manip import load_data, test_file, class_dict
from data_manip import show_point_cloud_panda, show_point_clouds, load_points_from_stl, draw_registration_result
import numpy as np
from keras_model_class import Keras_Custom_Model

import matplotlib.pyplot as plt

def plot_matplotlib(test_object, pred):
    # Create a new plot
    figure = plt.figure(f'3D Model')
    ax = figure.add_subplot(projection='3d')

    background_color = "#FF69B4"
    object_color = "#0096FF"
    print(len(test_object[0]))
    print(pred)
    for i in range(len(test_object[0])):
        if pred[i] == 0:
            ax.scatter(test_object[0][i][0], test_object[0][i][1], test_object[0][i][2], color=object_color)
        else:
            ax.scatter(test_object[0][i][0], test_object[0][i][1], test_object[0][i][2], color=background_color)
    plt.show()

# load data
test_data, test_labels, test_mask = load_data(test_file)




# load model
model = Keras_Custom_Model(model=None, model_name="fcnn_custom")
model.load_keras_model("models/keras_checkpoint_fully_connected_nn_custom_layers.keras")
# model.load_keras_model("models/keras_checkpoint_cnn.keras")

# try stl file
# point_cloud = load_points_from_stl("data/eot_metal.stl", True)
point_cloud = load_points_from_stl("data/tape.stl", True)

indices = np.array(range(len(point_cloud)))
indices = np.random.choice(indices, 2048, False)
object_model = point_cloud[indices]
# model_subset = np.expand_dims(model_subset, 0)
print(f"from {point_cloud.shape} to {object_model.shape}")

# index = [5, 200, 555, 999]
# for i in index:
#     object_model = test_data[i]
#     object_bg_mask = test_mask[i]
#     object_label = class_dict[f'{test_labels[i]}']

#     print(f'Object is part of a {object_label}.')

#     # run through model
#     object_model = np.expand_dims(object_model, 0)      # give it a number of objects of 1
#     prediction = model.predict_model(object_model)      # perform prediction
#     # print(f'pred shape: {prediction.shape}')
#     # print(prediction)
#     # print((prediction==1).sum())
#     # print(np.unique(prediction))
#     object_model = np.reshape(object_model, (2048, 3))  # get rid of number of objects

#     acc = (object_bg_mask==prediction).sum()            # count accuracy of prediction
#     acc = acc / object_bg_mask.shape[0]
#     print(f"accuracy for object: {acc:.2f}")

#     # print(object_model.shape)

#     # break into background and foreground
#     background_object = object_model[prediction==1]
#     foreground_object = object_model[prediction!=1]

#     # ground truth
#     truth_bg = object_model[object_bg_mask==1]
#     truth_fg = object_model[object_bg_mask==0]
#     # print(truth_bg.shape, truth_fg.shape)

#     actual_background_points = (object_bg_mask==1).sum()
#     pretend_accuracy = (2048-actual_background_points)/2048
#     print(f'accuracy if every point is foreground: {pretend_accuracy:.2f}')
#     # print(f'background objects: {(object_bg_mask==1).sum()}')
#     # print(f'unique: {np.unique(object_bg_mask)}')

#     background_object = np.resize(background_object, (2048, 3))
#     foreground_object = np.resize(foreground_object, (2048, 3))
#     truth_bg = np.resize(truth_bg, (2048, 3))
#     truth_fg = np.resize(truth_fg, (2048, 3))

#     # print('predicted shapes', background_object.shape, foreground_object.shape)
#     # print('ground truth shapes', truth_bg.shape, truth_fg.shape)

#     # show ground truth
#     show_point_clouds([truth_bg, truth_fg])
#     # show_point_clouds([point_cloud])

#     # show prediction
#     show_point_clouds([background_object, foreground_object])


