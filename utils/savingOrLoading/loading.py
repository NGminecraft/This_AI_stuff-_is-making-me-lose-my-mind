import pickle
import logging
import keras
import pandas as pd
import numpy as np
import os

class file_loader:
    def __init__(self, logger=None):
        if logger != None:
            self.logger = logger
            self.should_log = True
            self.logger.log(logging.INFO, "Activated file loader's logger")
        else:
            self.should_log = False
        self.cache = {}
    
    def load(self, path, cache_override=False):
        if path not in self.cache and not cache_override:
            self.logger.log(logging.INFO, f'Loading file: {path}')
            try:
                with open(path, "rb") as file:
                    a= pickle.load(file)
                    self.cache[path] = a
                    return a
            except EOFError:
                return None
        else:
            return self.cache[path]
    
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
    
    def exists(self, path):
        return os.path.exists(path)
    