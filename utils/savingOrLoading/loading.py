import pickle
from logging import INFO, WARN, ERROR
import keras
import pandas as pd
import numpy as np
import os

class FileLoader:
    def __init__(self, logger=None):
        if logger is not None:
            self.logger = logger
            self.should_log = True
            self.logger.log(INFO, "Activated file loader's logger")
        else:
            self.should_log = False
        self.cache = {}
    
    def load(self, path, cache_override=False):
        if path not in self.cache and not cache_override:
            self.logger.log(INFO, f'Loading file: {path}')
            try:
                with open(path, "rb") as file:
                    a= pickle.load(file)
                    self.cache[path] = a
                    return a
            except EOFError:
                self.logger.log(WARN, f'Attempted to read file at {path}, but it was empty!')
                return None
        else:
            return self.cache[path]
    
    def load_csv(self, csv):
        if csv in self.cache:
            return self.cache[csv]
        else:
            self.cache[csv] = pd.read_csv(csv)
            return self.cache[csv]
        
    def load_model(self, directory):
        try:
            return keras.saving.load_model(directory)
        except FileNotFoundError:
            if self.should_log:
                self.logger.log(ERROR, f'Attempted to load nonexistent model from {directory}')
            return None
    
    def load_numpy(self, npy):
        if npy[-3:] == 'npy':
            self.cache[npy] = np.load(npy)
            return self.cache[npy]
        else:
            if self.should_log:
                self.logger.log(ERROR, f'Attempted to read an invalid npy file at {npy}')
            return None

    @staticmethod
    def exists(path):
        return os.path.exists(path)
    