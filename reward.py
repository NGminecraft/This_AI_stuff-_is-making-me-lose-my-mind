import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import pandas as pd
import tensorflow as tf

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
        self.model, built = init_model()
        self.init_train_data()
        self.init_train_test()
        if not built:
            self.build_model()
            #            keras.saving.save_model(self.model, 'data//reward//models//model.keras')

    def call(self, inputs, training=False):
        return self.model(inputs)

    def train_step(self, model, inputs, labels, optimizer, loss_fn):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def init_train_data(self):
        self.data_train = load_data("data_train.csv")

    def init_train_test(self):
        self.data_test = load_data("data_test.csv")

    def build_model(self):
        activations = ["relu", "tanh", "sigmoid", "softmax", "softplus", "relu6", "softsign", "selu", "elu",
                       "exponential", "leaky_relu", "silu", "gelu", "hard_sigmoid", "linear", "mish", "log_softmax"]
        losses = ["binary_crossentropy", "categorical_crossentropy", "sparse_categorical_crossentropy", "poisson",
                  "kl_divergence", "mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error",
                  "mean_squared_logarithmic_error", "cosine_similarity", "huber", "log_cosh", "hinge", "squared_hinge",
                  "categorical_hinge"]
        
        for i in activations:
            self.model.add(keras.layers.Hashing(num_bins=len(self.data_train["Emotion"]), output_mode="one_hot"))
            self.model.add(keras.layers.Reshape((7934, 1)))
            self.model.add(keras.layers.Conv1D(64, 5, activation =i))
            self.model.add(keras.layers.Dense(5, activation =i))
            self.model.add(keras.layers.Dense(3, activation =i))
            self.model.add(keras.layers.Dense(1, activation=i))
            self.model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=["accuracy"])
            self.train()
            print(self.model.evaluate(x=self.data_train["Text"].to_list(), y=self.data_train["Emotion"].to_list(), batch_size=100))

    def train(self):
        self.model.fit(x=self.data_train["Text"].to_list(), y=self.data_train["Emotion"].to_list(), batch_size=100,
                       epochs=1,
                       validation_data=(self.data_test["Text"].to_list(), self.data_test["Emotion"].to_list()))
        keras.saving.save_model(self.model, "data//reward/models/trained.keras")


print("hello?")

reward = Reward()
