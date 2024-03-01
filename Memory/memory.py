import numpy as np
import os
import pickle
import threading
import logging
from Memory.object_handler import Handler as oh

class Memory:
    def __init__(self, exceptions, logger = None, file="Memory/Data/Memory.npy", should_log = True, module_loader = None):
        if exceptions is not None:
            self.exceptions = exceptions
        if logger is not None:
            self.logger = logger
            self.should_log = should_log
        if os.path.exists(file):
            self.memory_arr = pickle.load(file)
        else:
            self.build_memory(file)
        if module_loader:
            self.object_handler = module_loader.load(oh)
        else:
            self.object_handler = oh()
        if self.should_log:
            self.logger.log(logging.INFO, 'Sucsefully loaded Memory')
            
    def build_memory(self, file="Memory/Data/Memory.npy"):
        self.object_dict = {}
        
    def new_word(self, word:list|str):
        if type(word) is list:
            for i in word:
                pass