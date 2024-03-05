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
        
        if os.path.exists(path+"/mem_model.keras"):
            self.logger.log(logging.INFO, 'Found model, loading')
            self.model = keras.saving.load_model(path+"/mem_model.keras")
        else:
            self.logger.log(logging.INFO, 'No model found, building a new one')
            self.first_build(path)
    @staticmethod
    def _build(self):
        layers_obj = keras.layers
        word_input = layers_obj.Input((20, ))
        sentence_input = layers_obj.Input((500,))

        word_mask = layers_obj.Masking(mask_value=0)(word_input)
        lstm_word = layers_obj.LSTM(activation='tanh', recurrent_activation='sigmoid', return_sequences=True)(word_mask)

        sentence_mask = layers_obj.Masking(mask_value=0)(sentence_input)
        lstm_sentence = layers_obj.LSTM(activation='tanh', recurrent_activation='sigmoid', return_sequences=True)(sentence_mask)

        adder = layers_obj.add([lstm_word, lstm_sentence])

        dense_1 = layers_obj.Dense(units=20, activation='relu')(adder)
        dense_2 = layers_obj.Dense(units=10, activation='relu')(dense_1)
        output = layers_obj.Dense(1, activation='linear')(dense_2)

        return keras.Model(inputs=[word_input, sentence_input], outputs=output)
        
            
    def first_build(self, path):
        self.model = self._build()
        self.logger.log(logging.INFO, 'Built Memory Model')
        keras.saving.save_model(self.model, path+"/mem_model")