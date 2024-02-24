import keras
import pandas as pd
import os
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

def format(word):
    pass

def init_model():
    if os.path.exists('data/reward/models/model.keras'):
        return keras.saving.load_model('data//reward//models//model.keras'), True
    else:
        return keras.Sequential(), False


def load_csv(csv):
    return pd.read_csv(csv)


def load_data(file):
    data = load_csv(file)
    words_to_numbers = {
        "neutral": 0,
        "sadness": -1,
        "fear": -2,
        "anger": -3,
        "joy": 1
    }
    for i, v in enumerate(data["Emotion"]):
        data.at[i, "Emotion"] = words_to_numbers[v]
    return data


class Reward():
    def __init__(self, classes = 5, logger=None, str_obj=None):
        self.num_classes = classes
        self.logger = logger
        self.init_train_data()
        self.init_train_test()
        self.learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
        self.optimizer_names = dir(keras.optimizers)
        self.optimizers = ['Adadelta','Adagrad','Adam','Adamax','Ftrl','Nadam','RMSprop','SGD']
        self.losses = ['MeanSquaredError', 'MeanAbsoluteError', 'MeanAbsolutePercentageError',
                       'MeanAbsoluteLogarithmicError', 'CosineSimilarity', 'Huber',
                       'LogCosh', 'BinaryCrossentropy', 'CategoricalCrossentropy',
                       'SparseCategoricalCrossentropy', 'Hinge', 'SquaredHinge',
                       'CategoricalHinge', 'Poisson', 'KLDivergence', 'SquaredError']
        self.activations = ["softmax","softplus",'softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
        self.best_optimizer = None
        self.tuner = kt.RandomSearch(
            self._model_build,
            objective='val_loss',
            max_trials = 1000,
            directory = 'Reward/Data/Models',
            project_name='Reward Model',
            logger = self.logger
        )
        if str_obj is not None:
            self.tokenizer = str_obj.hash_text
            self.padding = str_obj.padding
            self.vocab = str_obj.get_vocabulary
        else:
            self._tokenizer = keras.preprocessing.text.Tokenizer(num_words=100000, oov_token=-1, split=' ')
            b = copy.deepcopy(self.data_train)["Text"].to_list()
            b.extend(self.data_test["Text"].to_list())
            self._tokenizer.fit_on_texts(b)
            self.tokenizer = self._tokenizer.texts_to_sequences
            self.padding = keras.preprocessing.sequence.pad_sequences
        self.model, built = init_model()
        if not built:
            self.build_model()
    
    def init_train_data(self):
        self.data_train = load_csv("Reward/Data/TrainingData/data_train.csv")

    def init_train_test(self):
        self.data_test = load_csv("Reward/Data/TrainingData/data_test.csv")
        
    def _model_build(self, hp):
        word_input = keras.layers.Input(shape=(300,), name="Word_Input")
        additional_input = keras.layers.Input(shape=(1,), name='additional_input')

        reshaped_input = keras.layers.Reshape((1, 300))(word_input)
        
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


    def build_model(self):

        if os.path.exists("Reward/Data/TrainingData/InputsandOutputs/inputs.npy") and os.path.exists("Reward/Data/TrainingData/InputsandOutputs/outputs.npy") and os.path.exists("Reward/Data/TrainingData/InputsandOutputs/validation_inputs.npy") and os.path.exists("Reward/Data/TrainingData/InputsandOutputs/validation_outputs.npy"):
            self.logger.log(logging.INFO, "Loading data from the npy files")
            input_indices = np.load("Reward/Data/TrainingData/InputsandOutputs/inputs.npy")
            self.logger.log(logging.INFO, f"Loaded input matrix of shape {input_indices.shape}")
            output_indices = np.load("Reward/Data/TrainingData/InputsandOutputs/outputs.npy")
            self.logger.log(logging.INFO, f"Loaded output matrix of shape {output_indices.shape}")
            validation_input_indices = np.load("Reward/Data/TrainingData/InputsandOutputs/validation_inputs.npy")
            self.logger.log(logging.INFO, f"Loaded validation input matrix of shape {validation_input_indices.shape}")
            validation_output_indices = np.load("Reward/Data/TrainingData/InputsandOutputs/validation_outputs.npy")
            self.logger.log(logging.INFO, f"Loaded validation output matrix of shape {validation_output_indices.shape}")
        else:
            self.logger.log(logging.INFO, "Missing, regenerating training data")
            input_length = 300

            words_to_numbers = {
                "neutral": 0,
                "sadness": -1,
                "fear": -2,
                "anger": -3,
                "joy": 1
            }
            for i, v in enumerate(self.data_train["Emotion"]):
                self.data_train.at[i, "Emotion"] = words_to_numbers[v]
            for i, v in enumerate(self.data_test["Emotion"]):
                self.data_test.at[i, "Emotion"] = words_to_numbers[v]
                
            input_indices = np.array([0]*input_length)
            for i in tqdm(self.data_train["Text"], desc="Creating The Input Matrix"):
                i = [i]
                if sys.getsizeof(input_indices) >= 1000000000:
                    self.logger.log(logging.WARNING, f'Input matrix is larger then 1 GB. Size = {sys.getsizeof(input_indices)}')
                input_indices = np.vstack((input_indices, self.padding(self.tokenizer(i), maxlen=input_length, padding='post', truncating='pre')))
            np.save("Reward/Data/TrainingData/InputsandOutputs/inputs", input_indices, allow_pickle=True)
            output_indices = np.array([[0]])
            for i in tqdm(self.data_train["Emotion"], desc="Creating The Output Matrix"):
                output_indices = np.vstack((output_indices, [i]))
            np.save("Reward/Data/TrainingData/InputsandOutputs/outputs", output_indices, allow_pickle=True)


            validation_input_indices = np.array([0]*input_length)
            for i in tqdm(self.data_test["Text"], desc="Creating The Validation Input Matrix"):
                i = [i]
                validation_input_indices = np.vstack((validation_input_indices, self.padding(self.tokenizer(i), maxlen=input_length, padding='post', truncating='post')))
            np.save("Reward/Data/TrainingData/InputsandOutputs/validation_inputs", validation_input_indices, allow_pickle=True)
            validation_output_indices = np.array([[0]])
            for i in tqdm(self.data_test["Emotion"], desc="Creating The Validation Output Matrix"):
                validation_output_indices = np.vstack((validation_output_indices, np.asarray([i])))
            np.save("Reward/Data/TrainingData/InputsandOutputs/validation_outputs", validation_output_indices,
                    allow_pickle=True)
            
        thingy = np.asarray([1]*input_indices.shape[0])
        val_thingy = np.asarray([1]*validation_input_indices.shape[0])
        
        self.logger.log(logging.INFO, f'Input loaded with shape {input_indices.shape} and looks like {input_indices}')
        self.logger.log(logging.INFO, f'Output loaded with shape {output_indices.shape} and looks like {output_indices}')
        self.logger.log(logging.INFO, f'Validation Input loaded with shape {validation_input_indices.shape} and looks like {validation_input_indices}')
        self.logger.log(logging.INFO, f'Validation Output loaded with shape {validation_output_indices.shape} and looks like {validation_output_indices}')

        self.tuner.search(x=[input_indices, thingy], y=output_indices, epochs=100, validation_data=([validation_input_indices, val_thingy], validation_output_indices))
        best_hps = self.tuner.get_best_models(num_models=3)
        for i in best_hps:
            i.save(f'Reward/Data/Models/best_hp-{i}')