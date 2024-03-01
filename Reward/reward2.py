import keras
import pandas as pd
import os

import utils.inputs_preparation.formatter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import logging
import copy
import sys
import matplotlib.pyplot as plt
from random import uniform
import keras_tuner as kt
import inspect


class Reward:
    def __init__(self, loader, formatter:utils.inputs_preparation.formatter, classes = 5, logger=None, build_model=True, exceptions=None):
        self.num_classes = classes
        self.logger = logger
        self.loader = loader
        all_training_text = self.loader.load_csv("Reward/Data/TrainingData/data_test.csv")["Text"].tolist()
        all_training_text.extend(self.loader.load_csv("Reward/Data/TrainingData/data_train.csv")["Text"].tolist())
        self.formatter = formatter(logger=logger, starting_text=all_training_text)
        self.model, built = self.init_model()
        
        if not built and build_model and self.model is None:
            self.train(find_best=True)
        self.logger.log(logging.INFO, 'Initialized reward Succsefully')
        
    def train(self, find_best=False, model=None):
        if find_best:
            self.learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
            self.optimizers = ['Adadelta','Adagrad','Adam','Adamax','Ftrl','Nadam','RMSprop','SGD']
            self.losses = ['MeanSquaredError', 'MeanAbsoluteError', 'MeanAbsolutePercentageError', 'Huber',
                        'LogCosh', 'BinaryCrossentropy', 'CategoricalCrossentropy',
                            'Hinge', 'SquaredHinge',
                        'CategoricalHinge', 'KLDivergence', 'SquaredError']
            self.activations = ["softmax","softplus",'softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
            self.tuner2 = kt.BayesianOptimization(
                self._model_build,
                objective='val_mean_squared_error',
                max_trials = 150,
                directory = 'Reward/Data/Models',
                project_name='Retrying the training',
                logger = self.logger,
                overwrite=False
            )
        self.build_model(find_best, model)
        
    def _model_build(self, hp):
        word_input = keras.layers.Input(shape=(500,), name="Word_Input")
        additional_input = keras.layers.Input(shape=(1,), name='additional_input')

        reshaped_input = keras.layers.Reshape((1, 500))(word_input)
        
        dropout = keras.layers.Dropout(1/3)(reshaped_input)

        lstm_layer = keras.layers.LSTM(units=hp.Int('LSTM units', min_value=10, max_value=1000, step=10), input_shape=(None, 1), activation=hp.Choice('LSTM_act', self.activations))(dropout)
        
        concatenated = keras.layers.Concatenate()([lstm_layer, additional_input])
        
        dense_layer_1 = keras.layers.Dense(units=hp.Int('dense_unit_1', min_value=10, max_value=1000, step=5),activation=hp.Choice('dense_1', self.activations))(concatenated)
        dense_layer_2 = keras.layers.Dense(units=hp.Int('dense_unit_2', min_value=10, max_value=1000, step=5),activation=hp.Choice('dense_2', self.activations))(dense_layer_1)
        dense_layer_3 = keras.layers.Dense(units=hp.Int('dense_unit_3', min_value=10, max_value=1000, step=5), activation=hp.Choice('dense_3', self.activations))(dense_layer_2)
        dense_layer_4 = keras.layers.Dense(units=hp.Int('dense_unit_4', min_value=10, max_value=1000, step=5), activation=hp.Choice('dense_4', self.activations))(dense_layer_3)
        dense_layer_7 = keras.layers.Dense(units=hp.Int('dense_unit_7', min_value=10, max_value=1000, step=5), activation=hp.Choice('dense_7', self.activations))(dense_layer_4)
        output = keras.layers.Dense(units=1, activation='tanh', name='output')(dense_layer_7)

        optimizer = hp.Choice('optimizer', self.optimizers)
        loss = hp.Choice('loss', self.losses)
        learning_rate = hp.Choice('learning_rate', self.learning_rates)
        
        model = keras.Model(inputs=[word_input, additional_input], outputs=output)

        if loss == 'Log_Cosh':
            loss = 'LogCosh'
        model.compile(optimizer=optimizer, loss=loss, metrics=['mean_squared_error'])
        return model


    def build_model(self, find_best=False, model=None):

        if os.path.exists("Reward/Data/TrainingData/InputsandOutputs/inputs.npy") and os.path.exists("Reward/Data/TrainingData/InputsandOutputs/outputs.npy") and os.path.exists("Reward/Data/TrainingData/InputsandOutputs/validation_inputs.npy") and os.path.exists("Reward/Data/TrainingData/InputsandOutputs/validation_outputs.npy"):
            self.logger.log(logging.INFO, "Loading data from the npy files")
            input_indices = self.loader.load_numpy("Reward/Data/TrainingData/InputsandOutputs/inputs.npy")
            self.logger.log(logging.INFO, f"Loaded input matrix of shape {input_indices.shape}")
            output_indices = self.loader.load_numpy("Reward/Data/TrainingData/InputsandOutputs/outputs.npy")
            self.logger.log(logging.INFO, f"Loaded output matrix of shape {output_indices.shape}")
            val_input_indices = self.loader.load_numpy("Reward/Data/TrainingData/InputsandOutputs/validation_inputs.npy")
            self.logger.log(logging.INFO, f"Loaded validation input matrix of shape {val_input_indices.shape}")
            val_output_indices = self.loader.load_numpy("Reward/Data/TrainingData/InputsandOutputs/validation_outputs.npy")
            self.logger.log(logging.INFO, f"Loaded validation output matrix of shape {val_output_indices.shape}")
        else:
            stuffy = self.load_training_data()
            if not hasattr(self, 'input_indices'):
                self.input_indices = stuffy[0][0].astype('float32')
                self.output_indices = stuffy[0][1].astype('float32')
                self.val_input_indices = stuffy[1][0].astype('float32')
                self.val_output_indices = stuffy[1][1].astype('float32')
                self.thingy = np.asarray([1]*self.input_indices.shape[0]).astype('float32')
                self.val_thingy = np.asarray([1]*self.val_input_indices.shape[0]).astype('float32')
            self.input_indices.save("Reward/Data/InputsandOutputs/inputs.npy")
            self.input_indices.save("Reward/Data/InputsandOutputs/outputs.npy")
            self.input_indices.save("Reward/Data/InputsandOutputs/validation_inputs.npy")
            self.input_indices.save("Reward/Data/InputsandOutputs/validation_outputs.npy")
        
        self.logger.log(logging.INFO, f'Input loaded with shape {self.input_indices.shape} and looks like {self.input_indices}')
        self.logger.log(logging.INFO, f'Output loaded with shape {self.output_indices.shape} and looks like {self.output_indices}')
        self.logger.log(logging.INFO, f'Validation Input loaded with shape {self.val_input_indices.shape} and looks like {self.val_input_indices}')
        self.logger.log(logging.INFO, f'Validation Output loaded with shape {self.val_output_indices.shape} and looks like {self.val_output_indices}')

        if find_best:
            self.tuner2.search(x=[self.input_indices, self.thingy], y=self.output_indices, epochs=100, validation_data=([self.val_input_indices, self.val_thingy], self.val_output_indices))
            best_hps = self.tuner2.get_best_models(num_models=3)
            for i in best_hps:
                i.save(f'Reward/Data/Models/best_hp-{i}')
        elif model is not None and not find_best:
            model.fit([self.input_indices, self.thingy], self.output_indices, epochs=100, valudation_data=([self.val_input_indices, self.val_thingy], self.val_output_indices))
        elif model is None:
            self.logger.log(logging.ERROR, f'No model found for training')


    def init_model(self):
        if not hasattr(self, 'input_indices'):
            data = self.load_training_data()
            self.input_indices = data[0][0].astype('float32')
            self.output_indices = data[0][1].astype('float32')
            self.val_input_indices = data[1][0].astype('float32')
            self.val_output_indices = data[1][1].astype('float32')
            self.thingy = np.asarray([1]*self.input_indices.shape[0]).astype('float32')
            self.val_thingy = np.asarray([1]*self.val_input_indices.shape[0]).astype('float32')
        tensorflow_items = [i for i in os.listdir("Reward/Data/Models") if "best_hp-" in i]
        if len(tensorflow_items) > 1:
            models = {}
            for i in tensorflow_items:
                model = self.loader.load_model(os.path.join("Reward/Data/Models", i))
                models[model.evaluate(x=[self.input_indices, self.thingy], y=self.output_indices)] = model
            return models[sorted(models.keys())[0]], True
        elif len(tensorflow_items) == 1:
            return self.loader.load_model(os.path.join("Reward/Data/Models", tensorflow_items[0])), True
        else:
            self.logger.log(logging.INFO, "No model found, creating one")
            return None, False

        
    def load_training_data(self) -> np.array:
        words_to_numbers = {
            "neutral": 0,
            "sadness": -1,
            "fear": -1,
            "anger": -1,
            "joy": 1
        }
        all_data = []
        for i, v in enumerate(["Reward/Data/TrainingData/data_train.csv", "Reward/Data/TrainingData/data_test.csv"]):
            data = self.loader.load_csv(v)
            for index, value in enumerate(data["Emotion"]):
                data.at[index, "Emotion"] = words_to_numbers[value]
            all_data.append([self.formatter.format(data["Text"]), data["Emotion"]])
        return all_data