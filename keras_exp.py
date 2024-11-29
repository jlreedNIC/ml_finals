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
from  other_libraries_used.pointnet1_keras.pointnet import PointNetFull
from other_libraries_used.pointconvTF2.model_modelnet import PointConvModel

# import model class
from keras_model_class import Keras_Custom_Model

# import data manipulation functions
from data_manip import load_data, train_file, test_file

# other imports
import keras
import os

def load_all_data():
    train_data, train_label, train_mask = load_data(train_file)
    test_data, test_label, test_mask = load_data(test_file)

    return train_data, train_label, train_mask, test_data, test_label, test_mask

def build_pointnet_model(filepath):
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


# ------ run experiments -------
train_data, train_label, train_mask, test_data, test_label, test_mask = load_all_data()

model = Keras_Custom_Model(build_pointnet_model(), "keras_pointnet")

# model = build_pointnet_model()
# model = build_pointconv_model()
model.build_callbacks()


batch_sizes = [16]#, 32, 64]
epochs = [10, 50, 100]
optimizer = ['adam', 'adamw']
validation_split = [0, .1]#, .2, .3]

for batch in batch_sizes:
    # model = build_pointconv_model(batch, (train_data[0].shape))
    for epoch in epochs:
        for opt in optimizer:
            # compile model
            # model = compile_model(model, opt)
            model.compile_model(opt)

            for valid in validation_split:
                model_name = f"pointconv_b{batch}_e{epoch}_o{opt}_v{int(valid*100)}"
                params = ['batch', batch, 'epochs', epoch, 'optimizer', opt, 'validation', valid]
                print(f'\nNow running model: {model_name}')

                # train model
                model, history = model.train_model(train_data, train_mask, batch, epoch, valid)
                # score model
                scores = model.score_model(train_data, train_mask, test_data, test_mask)

                # save data
                save_data(f'exp1/{model_name}.csv', model_name, scores, params)


# model, history = train_model(model, train_data, train_mask, 16, 10, callbacks)
# train_score, test_score = score_model(model, train_data, train_mask, test_data, test_mask)
# save_data('final_project_')

# callback_funcs = build_callbacks()

# # build model
# model = PointNetFull(num_points=2048, num_classes=16)
# model.compile(optimizer='adam',
#                 loss='sparse_categorical_crossentropy',
#                 metrics=['accuracy'])

# # load data
# train_data, train_label, train_mask = load_data(train_file)
# test_data, test_label, test_mask = load_data(test_file)

# # train model
# history = model.fit(train_data, train_mask, batch_size=16, epochs=10, callbacks=callback_funcs)

# # evaluation model
# train_score = model.evaluate(x=train_data, y=train_label, verbose=0)
# test_score = model.evaluate(x=test_data, y=test_label, verbose=0)

# # print score
# print(f'Train score: {train_score}')
# print(f'Test score: {test_score}')
