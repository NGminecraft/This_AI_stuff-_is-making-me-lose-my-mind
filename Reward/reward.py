import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import pandas as pd
import tensorflow as tf
import copy
import numpy as np

print(f"Starting with tensorflow version: {tf.__version__}")
if tf.test.is_gpu_available:
    print("GPU available")
    print(tf.test.gpu_device_name())
else:
    print("GPU not available")


def init_model():
    if os.path.exists('data//reward//models//model.keras'):
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


class Reward(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.tokenizer = keras.preprocessing.text.Tokenizer(num_words=1000,  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,  split=' ', char_level=False)
        self.model, built = init_model()
        self.init_train_data()
        self.init_train_test()
        if not built:
            self.build_model()
            #            keras.saving.save_model(self.model, 'data//reward//models//model.keras')

    def call(self, inputs, training=False):
        outputs = []
        prev_val = 0
        print(inputs)
        for i in inputs:
            print((1, i, prev_val))
            outputs.append(self.train_model(1, i, prev_val))
            print(prev_val)
            prev_val = outputs
        print(outputs)
        return self.train_model(inputs)

    def train_step(self, model, inputs, labels, optimizer, loss_fn):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def init_train_data(self):
        self.data_train = load_data("Data/TrainingData/data_train.csv")

    def init_train_test(self):
        self.data_test = load_data("Data/TrainingData/data_test.csv")

    def train(self, model=None, save=False, id="trained"):
        if model:
            use_model = model
        else:
            use_model = self.model
        # We need to loop through all of our data and then each word in the data, pass them into the NN and train with that.
        self.train_model = use_model
        self.tokenizer.fit_on_texts(self.data_train["Text"].to_list())
        inputs = self.tokenizer.texts_to_sequences(self.data_train["Text"].to_list())
        max_length = max(len(seq) for seq in inputs)
        inputs = keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_length)
        inputs = tf.cast(inputs, dtype=tf.float32)
        emotions = np.array(self.data_train["Emotion"].values.astype(int))
        use_model.fit(x=inputs, y=emotions, batch_size=1, epochs=1)
        if save == True:
            self.save(use_model, id=id)
            
    def save(self, m, id="trained"):
        if m:
            model = m
        else:
            model = self.model
        keras.saving.save_model(model, f"data//reward/models/{id}.keras")

    def build_model(self):
        activations = ["relu", "tanh", "sigmoid", "softmax", "softplus", "softsign", "selu", "elu",
                       "exponential", "gelu", "hard_sigmoid", "linear", "mish"]
        losses = ["binary_crossentropy", "categorical_crossentropy", "sparse_categorical_crossentropy", "poisson",
                  "kl_divergence", "mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error",
                  "mean_squared_logarithmic_error", "cosine_similarity", "huber", "log_cosh", "hinge", "squared_hinge",
                  "categorical_hinge"]
        for i, v in enumerate(activations):
            activations[i] = keras.activations.get(v)
        for i, v in enumerate(losses):
            losses[i] = keras.losses.get(v)
        best = 0
        best_loss = 0
        for j in losses:
            for i in activations:
                model = keras.Sequential()
                model.add(keras.Input(shape=(1, 188), name="Input", batch_size=1))
                model.add(keras.layers.Reshape((1, 188)))
                model.add(keras.layers.Conv1D(2, 1, activation =i))
                model.add(keras.layers.Flatten())
                model.add(keras.layers.Reshape((1 ,2)))
                model.add(keras.layers.Dense(2, activation=i))
                model.add(keras.layers.Dense(2, activation =i))
                model.add(keras.layers.Dense(2, activation =i))
                model.add(keras.layers.Dense(1, activation=i))
                model.compile(loss=j, optimizer='adam', metrics=["accuracy"])
                self.train(model)
                results = model.evaluate(x=self.data_train["Text"].to_list(), y=self.data_train["Emotion"].to_list(), batch_size=100)
                if results[1] > best:
                    best = results[1]
                    best_loss = results[0]
                    best_model = copy.deepcopy(model)
                elif results[1] == best:
                    if results[0] < best_loss:
                        best_loss = results[0]
                        best=results[1]
                        best_model = copy.deepcopy(model)
        self.save(model, "best1Activation")
        


reward = Reward()
