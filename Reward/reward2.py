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


class Reward:
    def __init__(self, logger=None, str_obj=None):
        self.logger = logger
        self.init_train_data()
        self.init_train_test()
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
        
    def _model_build(self, opt=keras.optimizers.Adam()):
        word_input = keras.layers.Input(shape=(300,), name="Word_Input")
        additional_input = keras.layers.Input(shape=(1,), name='additional_input')

        reshaped_input = keras.layers.Reshape((1, 300))(word_input)

        lstm_layer = keras.layers.LSTM(units=250, input_shape=(None, 1), activation='relu')(reshaped_input)
        
        concatenated = keras.layers.Concatenate()([lstm_layer, additional_input])
        
        dense_layer_1 = keras.layers.Dense(units=200, activation='linear')(concatenated)
        dense_layer_2 = keras.layers.Dense(units=100, activation='linear')(dense_layer_1)
        output = keras.layers.Dense(units=1, activation='tanh', name='output')(dense_layer_2)
        
        model = keras.Model(inputs=[word_input, additional_input], outputs=output)
        
        model.compile(optimizer=opt, loss='mse', metrics=['mean_squared_error'])
        
        self.logger.log(logging.INFO, 'model created')
        model.summary()
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
        
        self.logger.log(logging.INFO, f'Input loaded with shape {input_indices.shape} and looks like {input_indices}')
        self.logger.log(logging.INFO, f'Output loaded with shape {output_indices.shape} and looks like {output_indices}')
        self.logger.log(logging.INFO, f'Validation Input loaded with shape {validation_input_indices.shape} and looks like {validation_input_indices}')
        self.logger.log(logging.INFO, f'Validation Output loaded with shape {validation_output_indices.shape} and looks like {validation_output_indices}')

        optimizers = [keras.optimizers.SGD(), keras.optimizers.RMSprop(), keras.optimizers.Adadelta(), keras.optimizers.Adagrad(), keras.optimizers.Adam(), keras.optimizers.Adamax(), keras.optimizers.AdamW()]

        lowest_loss = 150
        best_optimizer = None
        best_history = None
        all_histories = []
        for i in optimizers:
            all_histories.append([i])
            model = self._model_build(opt = i)
            history = model.fit(x=[input_indices, thingy], y=output_indices, batch_size=10, epochs=20, validation_data=([validation_input_indices, np.asarray([1]*validation_input_indices.shape[0])], validation_output_indices))
            all_histories[-1].append(history)
            self.logger.log(logging.INFO, f'Trained model with optimizer: {i}')
            self.logger.log(logging.INFO, history)
            if history.history['val_loss'][-1] < lowest_loss:
                lowest_loss = history.history['val_loss'][-1]
                best_optimizer = None
                best_history = history
                self.logger.log(logging.INFO, f'{i} is the new best optimizer')
        x = np.arange(0, 20)
        for i in all_histories:
            history = i[1]
            plt.plot(x, history.history['loss'], label=f"Loss of {i[0].name}")
            plt.plot(x, history.history['val_loss'], label=f"Validation loss of {i[0].name}")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Model Comparison')
        plt.legend()
        plt.show()
        plt.savefig('ModelComparison.png')