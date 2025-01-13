# ------------------------
# @file     keras_model_class.py
# @date     November 29, 2024
# @author   Jordan Reed
# @email    reed5204@vandals.uidaho.edu
# @brief    model manipulation for final project
#           
# ------------------------

import keras
import numpy as np
import os

class Keras_Custom_Model():
    def __init__(self, model, model_name:str):
        self.model = model
        self.model_name = model_name

        self.callback_funcs = []
        self.train_score = 0
        self.test_score = 0

    def load_keras_model(self, filename:str):
        """
        Load a keras model from a file

        :param filename: string containing path to keras model
        """
        if os.path.exists(filename):
            self.model = keras.models.load_model(filename)
        else:
            print(f"no model found at {filename}")
            self.model = None

    def build_callbacks(self, checkpoint_file=None):
        """
        Implement early stopping and saving checkpoints for model

        :param checkpoint_file: filename to save model checkpoint to, defaults to None
        """
        if checkpoint_file is None:
            checkpoint_file = f"./model_checkpoints/keras_checkpoint_{self.model_name}.keras"
        else:
            checkpoint_file = f'./model_checkpoints/{checkpoint_file}.keras'
        
        monitoring = 'loss'
        callback_early_stopping = keras.callbacks.EarlyStopping(patience=2, monitor=monitoring)
        callback_checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_file, 
            monitor=monitoring, 
            mode='min', 
            save_best_only=True)
        
        self.callback_funcs = [callback_early_stopping, callback_checkpoint]

    def compile_model(self, optimizer='adam', loss=None, metrics=['accuracy']):
        """
        Compile model given parameters

        :param optimizer: optimizer to use with model, defaults to 'adam'
        :param loss: loss function, defaults to None
        :param metrics: list of metrics to track, defaults to ['accuracy']
        """
        if loss is None:
            loss = 'sparse_categorical_crossentropy'
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def train_model(self, train_data, train_labels, batch_size=16, epochs=100, validation=.1):
        """
        train a keras model

        :param train_data: training data, numpy list
        :param train_labels: training labels, numpy list
        :param batch_size: batch size, int, defaults to 16
        :param epochs: number of epochs to train for, int, defaults to 100
        :param validation: validation split to perform, between 0 and 1, defaults to .1
        :return: history object
        """
        history = self.model.fit(
            train_data, train_labels,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.callback_funcs,
            validation_split=validation,
        )

        return history

    def score_model(self, train_data, train_labels, test_data, test_labels):
        """
        evaluate the model's accuracy and loss on both train set and test set

        :param train_data: training data, numpy list
        :param train_labels: training labels, numpy list
        :param test_data: test data, numpy list
        :param test_labels: test labels, numpy list
        :return: train score loss and accuracy, test score loss and accuracy
        """
        self.train_score = self.model.evaluate(x=train_data, y=train_labels, verbose=0)
        self.test_score = self.model.evaluate(x=test_data, y=test_labels, verbose=0)

        print(f"training score: {self.train_score}")
        print(f'test score: {self.test_score}')

        return self.train_score, self.test_score

    def save_model(self):
        """
        save model to a file
        """
        self.model.save(f'models/{self.model_name}.keras')
    
    def predict_model(self, data):
        """
        perform a prediction for a single object

        :param data: object data of shape (1, 2048, 3)
        :return: list of predictions in shape (2048,)
        """
        predictions = self.model.predict(data)
        print(f'pred shape: {predictions.shape}')
        
        if "pointnet" in self.model_name:
            # -----------
            # predictions in pointnet
            # shape is (batch, 2048, 16)
            predictions = np.reshape(predictions, (2048, 16))
            predictions = np.argmax(predictions, axis=-1)
            print(f'pred shape: {predictions.shape}')
            # print(predictions)
            # print('num backgrounds', (predictions==1).sum())
            # ----------
        elif "fcnn" in self.model_name:
            # ---------
            # predictions for fcnn
            # shape (batch, 2048, 2)
            predictions = np.reshape(predictions, (2048, 2))
            predictions = np.argmax(predictions, axis=-1)
            print(f'pred shape: {predictions.shape}')
            # ------------
        elif "cnn" in self.model_name:
            # ------------
            # for predictions in cnn
            predictions = np.reshape(predictions, (2048,2,2))

            pred = []
            for i in range(len(predictions)):
                # fg = (predictions[i][0][1] + predictions[i][1][1]) / 2
                # bg = (predictions[i][0][0] + predictions[i][1][0]) / 2
                fg = predictions[i][1][1]
                bg = predictions[i][1][0]
                p = np.argmax([bg, fg])
                pred.append(p)
                # break
            predictions = np.array(pred)
            # ---------
        else:
            print("Unknown model type to run predictions for.")
            predictions = np.argmax(predictions, axis=-1)
            print(f'pred shape: {predictions.shape}')

        
        print('num backgrounds in prediction', (predictions==1).sum())
        return predictions
    
    def custom_CategoricalCrossentropy(self, y_true, y_pred):
        """
        Not used. Attempt for cnn evaluation.

        :param y_true: _description_
        :param y_pred: _description_
        :return: _description_
        """
        # y true must be one hot encoded
        # y pred is of shape (batch, 2048, 2, 2)
        # y_true = y_true.numpy()
        y_true = y_true[:,:,0]
        # y_pred = y_pred.numpy()
        pred_prob = []
        for i in range(len(y_pred)):
            batch_prob = []
            for j in range(len(y_pred[0])):
                fg = y_pred[i][j][1][1]
                bg = y_pred[i][j][1][0]
                batch_prob.append([bg, fg])
            pred_prob.append(batch_prob)

        pred_prob = np.array(pred_prob)

        loss_fn = keras.losses.CategoricalCrossentropy()
        loss_val = loss_fn(y_true, pred_prob)

        return loss_val

