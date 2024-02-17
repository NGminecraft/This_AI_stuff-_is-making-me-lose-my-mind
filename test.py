import tensorflow as tf
import keras

a = keras.preprocessing.text.Tokenizer(num_words=10)
a.fit_on_texts(["Hi Nick", "Hi Cows"])
b = a.texts_to_sequences(["Hi Nick", "Hi Cows"])
print(a.texts_to_sequences(["Hi Nick", "Hi Cows"]))
print(b)