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

    def format(self, *args):
        if type(args[0]) is list:
            obj_size = len(args)
            item = [self.padder.pad(self.tokenizer.tokenize(i)) for i in args]
            if len(item) != obj_size and self.should_log:
                self.logger.log(logging.ERROR, f"Size Mismatch: {obj_size} vs {len(args)}")
            return np.asarray(item)
        else:
            item = self.padder.pad(self.tokenizer.tokenize(args))
            return np.asarray(item)