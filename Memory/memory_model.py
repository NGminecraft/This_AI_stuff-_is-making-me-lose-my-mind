import os
import keras
import logging
import numpy as np

class MemModel:
    def __init__(self, logger=None, exceptions=None, path="Memory/Data/", module_loader=None, formatter=None, saver=None):
        self.formatter = formatter
        if logger is not None:
            self.logger = logger
            self.should_log = True
        else:
            self.should_log = False
        self.saver = saver
        if exceptions is not None:
            self.exceptions = logger
            self.should_have_errors = True
        else:
            self.should_have_errors = False
        
        if os.path.exists(path+"/memory_model.keras"):
            self.model = keras.saving.load_model(path+"/memory_model.keras")
            if self.should_log:
                self.logger.log(logging.INFO, 'Found model, loading')
                self.logger.log(logging.INFO, '')
                self.model.summary(print_fn=self.logger.info)
                self.logger.log(logging.INFO, '')
        else:
            if self.should_log:
                self.logger.log(logging.INFO, 'No model found, building a new one')
            self.first_build(path)
        self.logger.log(logging.INFO, 'Loaded memory model, testing')
        self.create_value("TESTING", "This is a TESTING sentence")
        self.logger.log(logging.INFO, 'Testing Completed')
            
    def Test(self):
        self.logger.log(logging.INFO, 'Testing Memory Model')
        self.create_value("TESTING", "This is a TESTING sentence")
        self.logger.log(logging.INFO, 'Testing Completed')
            
    @staticmethod
    def _build():
        layers_obj = keras.layers
        word_input = layers_obj.Input((None,20))
        sentence_input = layers_obj.Input((None,500))

        word_mask = layers_obj.Masking(mask_value=0)(word_input)
        reshape = layers_obj.Reshape((1, 20))(word_mask)
        lstm_word = layers_obj.LSTM(activation='tanh', units=20, recurrent_activation='sigmoid', return_sequences=True)(reshape)

        sentence_mask = layers_obj.Masking(mask_value=0)(sentence_input)
        reshape = layers_obj.Reshape((1, 500))(sentence_mask)
        lstm_sentence = layers_obj.LSTM(activation='tanh', units=20, recurrent_activation='sigmoid', return_sequences=True)(reshape)

        adder = layers_obj.add([lstm_word, lstm_sentence])

        dense_1 = layers_obj.Dense(units=20, activation='relu')(adder)
        dense_2 = layers_obj.Dense(units=10, activation='relu')(dense_1)
        output = layers_obj.Dense(1, activation='linear')(dense_2)

        return keras.Model(inputs=[word_input, sentence_input], outputs=output)
            
    def first_build(self, path):
        model = self._build()
        model.compile(loss=keras.losses.binary_crossentropy, optimizer="Adamax", metrics=["Accuracy"])
        self.logger.log(logging.DEBUG, str(model))
        self.logger.log(logging.INFO, 'Built Memory Model')
        if self.saver is not None:
            self.saver.save_model(model, path+"/memory_model.keras")
        b = model.summary()
        print(b)

    def create_value(self, word:str, context:list|str) -> float:
        self.logger.log(logging.DEBUG, context)
        if type(context) is str and " " in context:
            context = context.split(" ")
        self.logger.log(logging.DEBUG, context)
        input_word = self.formatter.format(True, (1, 1, -1), word, length_override=20)
        input_sentence = self.formatter.format(False, (1, 1, -1), context)
        return self.model([input_word, input_sentence], training=False)
