from utils.inputs_preparation.padder import Padder
from utils.inputs_preparation.tokenizer_handler import TokenizerHandler
import numpy as np
import logging

class Formatter:
    def __init__(self, starting_text:list=None, tokenizer_handler=TokenizerHandler(), padder=Padder(), logger=None):
        if logger is not None:
            self.logger = logger
            self.should_log = True
        else:
            self.should_log = False
        self.logger.log(logging.DEBUG, starting_text)
        self.tokenizer = tokenizer_handler
        if starting_text is not None:
            self.logger.log(logging.INFO, "Received starting text")
            self.tokenizer.refit_tokenizer(starting_text)
        self.padder = padder

    def format(self, obj):
        if type(obj[0]) is list:
            item = np.asarray([self.padder.pad(self.tokenizer.tokenize(i)) for i in obj])
            return item
        else:
            item = self.padder.pad(self.tokenizer.tokenize(obj))
            return item

    def add_texts(self, texts:str):
        self.tokenizer.append_token([texts])