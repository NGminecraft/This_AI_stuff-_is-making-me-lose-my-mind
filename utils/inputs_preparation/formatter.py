from utils.inputs_preparation.padder import Padder
from utils.inputs_preparation.tokenizer_handler import TokenizerHandler
import numpy as np
import logging
import pandas

class Formatter:
    def __init__(self, tokenizer_handler=TokenizerHandler(), padder=Padder(), logger=None):
        if logger is not None:
            self.logger = logger
            self.should_log = True
        else:
            self.should_log = False
        self.tokenizer = tokenizer_handler
        self.padder = padder

    def format(self, wordList=False, shape=(1, 1, -1), *args, **kwargs) -> np.array:
        if type(args[0]) is list and not wordList:
            obj_size = len(args)
            item = [self.padder.pad(self.tokenizer.tokenize(i), **kwargs) for i in args]
            if len(item) != obj_size and self.should_log:
                self.logger.log(logging.ERROR, f"Size Mismatch: {obj_size} vs {len(args)}")
            if shape is not None:
                return np.reshape(np.asarray(item), shape)
            else:
                return np.asarray(item)
        elif wordList:
            args = [list(i) for i in args]
            items = np.asarray([self.padder.pad(k, **kwargs) for k in [[ord(str(j)) for j in i] for i in args]])
            if shape is not None:
                return np.reshape(items, shape)
            else:
                return items
        else:
            item = self.padder.pad(self.tokenizer.tokenize(args))
            if shape is not None:
                return np.reshape(np.asarray(item), shape)
            else:
                return np.asarray(item)