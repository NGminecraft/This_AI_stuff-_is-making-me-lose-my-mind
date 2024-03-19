import os
"""
THIS IS THE WORKING CODE FOR LOADING THE TF MODELS

import keras

b = keras.models.load_model('Reward/Data/Models/best_hp-number1')

b.summary()
"""
dictionary = {'one': 1, 'three': 3, 'two': 2}
b = ['one', 'two']
sorted_keys = sorted(b, key=lambda x: dictionary[x], reverse=True)
soreted_dict = {key: dictionary[key] for key in sorted_keys}
print(soreted_dict)