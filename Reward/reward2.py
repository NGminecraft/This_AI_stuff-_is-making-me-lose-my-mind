import keras
import pandas as pd
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import logging
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
        "fear": -1,
        "anger": -1,
        "joy": 1
    }
    for i, v in enumerate(data["Emotion"]):
        data.at[i, "Emotion"] = words_to_numbers[v]
    return data


class Reward:
    def __init__(self, str_obj, logger):
        self.logger = logger
        self.tokenizer = str_obj.hash_text
        self.padding = str_obj.padding
        self.vocab = str_obj.get_vocabulary
        self.model, built = init_model()
        self.init_train_data()
        self.init_train_test()
        if not built:
            self.build_model()
    
    def init_train_data(self):
        self.data_train = load_data("Reward/Data/TrainingData/data_train.csv")

    def init_train_test(self):
        self.data_test = load_data("Reward/Data/TrainingData/data_test.csv")
        
    def _model_build(self, vocab_size=100000):
        word_input = keras.layers.Input(shape=(1,), name="Word_Input")
        prev_word_input = keras.layers.Input(shape=(1,), name='prev_word_input')
        additional_input = keras.layers.Input(shape=(1,), name='additional_input')
        
        word_embedding = keras.layers.Dense(200)(word_input)
        
        lstm_units = 150
        lstm_layer = keras.layers.LSTM(units=150, input_shape=(None, 1500))(word_input)
        
        concatenated = keras.layers.Concatenate()([lstm_layer, prev_word_input, additional_input])
        
        dense_layer_1 = keras.layers.Dense(units=128, activation='relu')(concatenated)
        dense_layer_2 = keras.layers.Dense(units=64, activation='relu')(dense_layer_1)
        output = keras.layers.Dense(units=1, activation='tanh', name='output')(dense_layer_2)
        
        model = keras.Model(inputs=[word_input, prev_word_input, additional_input], outputs=output)
        
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        
        self.logger.log(logging.INFO, 'model created')
        model.summary()
        return model
    
    def build_model(self):

        if os.path.exists("Reward/Data/TrainingData/InputsandOutputs/inputs.npy") and os.path.exists("Reward/Data/TrainingData/InputsandOutputs/outputs.npy"):
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
            self.logger.log(logging.INFO, "No data found, generating training data")
            input_length = 1500
            train_data = pd.read_csv("Reward/Data/TrainingData/data_train.csv")
            test_data = pd.read_csv("Reward/Data/TrainingData/data_test.csv")
            
            words_to_numbers = {
                "neutral": 0,
                "sadness": -1,
                "fear": -1,
                "anger": -1,
                "joy": 1
            }
            for i, v in enumerate(train_data["Emotion"]):
                train_data.at[i, "Emotion"] = words_to_numbers[v]
            for i, v in enumerate(test_data["Emotion"]):
                test_data.at[i, "Emotion"] = words_to_numbers[v]
                
            input_indices = np.array([0]*input_length)
            for i in tqdm(train_data["Text"], desc="Creating The Input Matrix"):
                try:
                    input_indices = np.vstack((input_indices, self.padding(list(self.tokenizer(j) for j in i), input_length, 0)))
                except ValueError:
                    print(self.padding(list(self.tokenizer(j) for j in i), input_length, 0))
                    print(len(self.padding(list(self.tokenizer(j) for j in i), input_length, 0)))
            np.save("Reward/Data/TrainingData/InputsandOutputs/inputs", input_indices, allow_pickle=True)
            output_indices = np.array([[0]])
            for i in tqdm(train_data["Emotion"], desc="Creating The Output Matrix"):
                output_indices = np.vstack((output_indices, [i]))
            np.save("Reward/Data/TrainingData/InputsandOutputs/outputs", output_indices, allow_pickle=True)



            input_length = 1500
            validation_input_indices = np.array([0]*input_length)
            for i in tqdm(test_data["Text"], desc="Creating The Validation Input Matrix"):
                try:
                    validation_input_indices = np.vstack((validation_input_indices, self.padding(list(self.tokenizer(j) for j in i), input_length, 0)))
                except ValueError:
                    print(self.padding(list(self.tokenizer(j) for j in i), input_length, 0))
                    print(len(self.padding(list(self.tokenizer(j) for j in i), input_length, 0)))
            np.save("Reward/Data/TrainingData/InputsandOutputs/validation_inputs", validation_input_indices, allow_pickle=True)
            validation_output_indices = np.array([[0]])
            for i in tqdm(train_data["Emotion"], desc="Creating The Validation Output Matrix"):
                validation_output_indices = np.vstack((validation_output_indices, np.asarray([i])))
            np.save("Reward/Data/TrainingData/InputsandOutputs/validation_outputs", validation_output_indices,
                    allow_pickle=True)
            
        thingy = np.asarray([0]*input_indices.shape[0])
        
        self.logger.log(logging.INFO, f'Input loaded with shape {input_indices.shape} and looks like {input_indices}')
        self.logger.log(logging.INFO, f'Output loaded with shape {output_indices.shape} and looks like {output_indices}')
        
        model = self._model_build(len(self.vocab().keys()))

        model.fit(x=[input_indices, thingy, thingy], y=output_indices, validation_data=(validation_input_indices, validation_output_indices))
