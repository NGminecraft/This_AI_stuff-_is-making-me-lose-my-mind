import numpy as np
import os
import pickle
import threading
import logging
from Memory.object_handler import Handler as oh
from Memory.memory_model import MemModel

class Memory:
    def __init__(self, exceptions, saver, logger = None, file="Memory/Data/Memory.npy", should_log = True, module_loader = None):
        if exceptions is not None:
            self.exceptions = exceptions
        if logger is not None:
            self.logger = logger
            self.should_log = should_log
        if os.path.exists(file):
            with open(file, "rb") as f:
                self.memory_arr = pickle.load(f)
        else:
            self._build_memory()
        if module_loader:
            self.object_handler = module_loader.load(oh)
        else:
            self.object_handler = oh()
        if module_loader is not None:
            self.loader = module_loader
            self.model = module_loader.load(MemModel)
        if self.should_log:
            self.logger.log(logging.INFO, 'Successfully loaded Memory')
        self.file_saver = saver
            
    def _build_memory(self):
        self.object_dict = {}
        
    def new_word(self, word:list|str):
        if type(word) is list:
            for i in word:
                self.object_dict[i] = self.model.create_value(i)
        elif type(word) is str:
            self.object_dict[i] = self.model.create_value(i)
            
    def save(self):
        self.file_saver.save_model(self.model, "Memory/Data/memory_model.keras", 'memory model')