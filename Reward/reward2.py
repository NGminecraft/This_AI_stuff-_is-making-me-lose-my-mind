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
    def __init__(self, str_obj):
        self.tokenizer = str_obj.hash_text
        self.padding = str_obj.padding
        self.model, built = init_model()
        self.init_train_data()
        self.init_train_test()
        if not built:
            self.build_model()
    
    def init_train_data(self):
        self.data_train = load_data("Reward/Data/TrainingData/data_train.csv")

    def init_train_test(self):
        self.data_test = load_data("Reward/Data/TrainingData/data_test.csv")
    
    def build_model(self):
        word_input = keras.layers.Input(shape=(1,), name="Word_Input")
        prev_word_input = keras.layers.Input(shape=(1,), name='prev_word_input')
        additional_input = keras.layers.Input(shape=(1,), name='additional_input')
        
        vocabulary_size = 12940
        embedding_dim = 200
        word_embedding = keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_dim)(word_input)
        
        lstm_units = 150
        lstm_layer = keras.layers.LSTM(units=lstm_units)(word_embedding)
        
        concatenated = keras.layers.Concatenate()([lstm_layer, prev_word_input, additional_input])
        
        dense_layer_1 = keras.layers.Dense(units=128, activation='relu')(concatenated)
        dense_layer_2 = keras.layers.Dense(units=64, activation='relu')(dense_layer_1)
        output = keras.layers.Dense(units=1, activation='tanh', name='output')(dense_layer_2)
        
        model = keras.Model(inputs=[word_input, prev_word_input, additional_input], outputs=output)
        
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        
        model.summary()

        if os.path.exists("Reward/Data/TrainingData/InputsandOutputs/inputs.npy") and os.path.exists("Reward/Data/TrainingData/InputsandOutputs/outputs.npy"):
            logging.log(logging.INFO, "Loading data from the npy files")
            input_indices = np.load("Reward/Data/TrainingData/InputsandOutputs/inputs.npy")
            logging.log(logging.INFO, f"Loaded input matrix of shape {input_indices.shape}")
            output_indices = np.load("Reward/Data/TrainingData/InputsandOutputs/outputs.npy")
            logging.log(logging.INFO, f"Loaded output matrix of shape {output_indices.shape}")
            validation_input_indices = np.load("Reward/Data/TrainingData/InputsandOutputs/validation_inputs.npy")
            logging.log(logging.INFO, f"Loaded validation input matrix of shape {validation_input_indices.shape}")
            validation_output_indices = np.load("Reward/Data/TrainingData/InputsandOutputs/validation_outputs.npy")
            logging.log(logging.INFO, f"Loaded validation output matrix of shape {validation_output_indices.shape}")
        else:
            input_length = 1500
            train_data = pd.read_csv("Reward/Data/TrainingData/data_train.csv")
            test_data = pd.read_csv("Reward/Data/TrainingData/data_test.csv")
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
                output_indices = np.vstack(i)
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
                validation_output_indices = np.vstack(i)
            np.save("Reward/Data/TrainingData/InputsandOutputs/validation_outputs", validation_output_indices,
                    allow_pickle=True)

        model.fit(x=[input_indices, 1, 1], y=output_indices, validation_data=(validation_input_indices, validation_output_indices))
