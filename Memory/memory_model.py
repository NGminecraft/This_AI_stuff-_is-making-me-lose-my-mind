import os
import keras
import logging

class MemModel:
    def __init__(self, logger=None, exceptions=None, path="Memory/Data/", module_loader=None, formatter=None):
        if logger is not None:
            self.logger = logger
            self.should_log = True
        else:
            self.should_log = False
        
        if exceptions is not None:
            self.exceptions = logger
            self.should_have_errors = True
        else:
            self.should_have_errors = False
        
        if os.path.exist(path+"/mem_model.keras"):
            self.logger.log(logging.INFO, 'Found model, loading')
            self.model = keras.saving.load_model(path+"/mem_model.keras")
        else:
            self.logger.log(logging.INFO, 'No model found, building a new one')
            self.first_build(path)
            
    def _build(self):
        layers_obj = keras.layers
        word_input = layers_obj.Input((20, ))
        sentence_input = layers_obj.Input((500,))
        
        lstm_word = layers_obj.LSTM
        
        
            
    def first_build(self, path):
        