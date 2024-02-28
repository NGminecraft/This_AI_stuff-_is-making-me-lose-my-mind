from utils.inputs_preparation.padder import Padder
from utils.inputs_preparation.tokenizer_handler import TokenizerHandler
import numpy as np

class Formatter:
    def __init__(self, starting_text:str=None, tokenizer_handler=TokenizerHandler(), padder=Padder()):
        self.tokenizer = tokenizer_handler
        if starting_text is not None:
            self.tokenizer.refit_tokenizer(starting_text)
        self.padder = padder

    def format(self, obj):
        if type(obj[0]) is list:
            return np.asarray([self.padder.pad(self.tokenizer.tokenize(i)) for i in obj])
        else:
            return self.padder.pad(self.tokenizer.tokenize(obj))

    def add_texts(self, texts:str):
        self.tokenizer.append_token([texts])