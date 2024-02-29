import pandas as pd
import os
import keras
import numpy as np
import logging

class Loader:
    """Loads stuff. May add saving stuff too."""
    def __init__(self, logger=None):
        self.cache = {}
        if logger != None:
            self.logger=logger
            self.should_log = True
            self.logger.log(logging.INFO, 'Activated logging for loading manager')
        else:
            self.should_log = False
    
    def load_csv(self, csv):
        if csv in self.cache:
            return self.cache[csv]
        else:
            self.cache[csv] = pd.read_csv(csv)
            return self.cache[csv]
        
    def load_model(self, dir):
        try:
            return keras.saving.load_model(dir)
        except FileNotFoundError:
            if self.should_log:
                self.logger.log(logging.ERROR, f'Attempted to load model from {dir}')
            return None
    
    def load_numpy(self, npy):
        if npy[-3:] == 'npy':
            self.cache[npy] = np.load(npy)
            return self.cache[npy]
        else:
            if self.should_log:
                self.logger.log(logging.WARNING, f'Attempted to read an invalid npy file at {npy}')
            return None