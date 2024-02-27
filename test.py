import os
"""
THIS IS THE WORKING CODE FOR LOADING THE TF MODELS

import keras

b = keras.models.load_model('Reward/Data/Models/best_hp-number1')

b.summary()
"""

dirs = [for i in os.listdir('Reward/Data/Models/Training attempt 2') if not os.path.isfile(os.path.join('Reward/Data/Models/Training attempt 2', i))]
