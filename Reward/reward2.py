import keras
import pandas as pd
import os
import tensorflow as tf
import numpy as np

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
    def __init__(self):
        self.tokenizer = keras.preprocessing.text.Tokenizer(num_words=1000,  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,  split=' ', char_level=False)
        self.model, built = init_model()
        self.init_train_data()
        self.init_train_test()
        if not built:
            self.build_model()
    
    def init_train_data(self):
        self.data_train = load_data("TrainingData/data_train.csv")

    def init_train_test(self):
        self.data_test = load_data("TraningData/data_test.csv")
    
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
        
        train_data = pd.read_csv("data_train.csv")
        test_data = pd.read_csv("data_test.csv")
        tokenizer_sentences = []
        for i in train_data["Text"]:
            tokenizer_sentences.append(i) 
        for i in test_data["Text"]:
            tokenizer_sentences.append(i)
        tokenizer = keras.preprocessing.text.Tokenizer(filters="():", lower=True, )
        tokenizer.fit_on_texts(tokenizer_sentences)
        c=[]
        for i in train_data["Text"]:
            b = []
            for j in i:
                if tokenizer.word_index.get(j.lower()) == None:
                    b.append(tokenizer.word_index.get(j.lower()))
            c.append(b)
        print(c)
        input_indices = np.asarray(c)
        model.fit(x=[input_indices, 1, 1], y=train_data["Emotion"], validation_data=[].append(tokenizer.word_index.get(i) for i in test_data))
        
        
Reward()
