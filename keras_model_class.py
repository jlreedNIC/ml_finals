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

class Keras_Custom_Model():
    def __init__(self, model, model_name:str):
        self.model = model
        self.model_name = model_name

        self.callback_funcs = []
        self.train_score = 0
        self.test_score = 0

    def build_callbacks(self, checkpoint_file=None):
        if checkpoint_file is None:
            checkpoint_file = f"./model_checkpoints/keras_checkpoint_{self.model_name}.keras"
        else:
            checkpoint_file = f'./model_checkpoints/{checkpoint_file}.keras'
        
        monitoring = 'loss'
        callback_early_stopping = keras.callbacks.EarlyStopping(patience=3, monitor=monitoring)
        callback_checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_file, 
            monitor=monitoring, 
            mode='min', 
            save_best_only=True)
        
        self.callback_funcs = [callback_early_stopping, callback_checkpoint]

    def compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def train_model(self, train_data, train_labels, batch_size=16, epochs=100, validation=.1):
        history = self.model.fit(
            train_data, train_labels,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.callback_funcs,
            validation_split=validation
        )

        return history

    def score_model(self, train_data, train_labels, test_data, test_labels):
        self.train_score = self.model.evaluate(x=train_data, y=train_labels, verbose=0)
        self.test_score = self.model.evaluate(x=test_data, y=test_labels, verbose=0)

        print(f"training score: {self.train_score}")
        print(f'test score: {self.test_score}')

        return self.train_score, self.test_score

    def save_model(self):
        self.model.save(f'models/{self.model_name}.keras')
    
    def predict_model(self, data):
        predictions = self.model.predict(data)
        print(f'pred shape: {predictions.shape}')
        predictions = np.argmax(predictions, axis=0)
        predictions = np.argmax(predictions, axis=1)
        print(f'pred shape: {predictions.shape}')

        return predictions

