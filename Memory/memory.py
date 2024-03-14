import numpy as np
import os
import pickle
import threading
import logging
from Memory.object_handler import Handler as oh
from Memory.memory_model import MemModel

class Memory:
    def __init__(self, exceptions, saver, logger = None, path="Memory/Data", should_log = True, module_loader = None):
        if exceptions is not None:
            self.exceptions = exceptions
        if logger is not None:
            self.logger = logger
            self.should_log = should_log
        if os.path.exists(path + "/Data/memory_model.keras"):
            with open(path + "/Data/memory_model.keras", "rb") as f:
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
        self._load_object_handler(path)
            
    def _build_memory(self):
        self.object_dict = {}
        
    def get_dict(self, key):
        self.logger.log(logging.DEBUG, key)
        if key in self.object_dict.keys():
            return self.object_dict[key]
        else:
            self.logger.log(logging.INFO, f'Word {key} not found. Creating.')
            self.new_word(key)
            
    def Test(self):
        self.model.Test()
        
    def _new_word(self, word:list|str):
        if type(word) is list:
            for i in word:
                self.object_dict[i] = self.model.create_value(i)
        elif type(word) is str:
            self.object_dict[i] = self.model.create_value(i)
            
    def memory_call(self, category=None, prompt=None):
        if category is None and prompt is None:
            raise self.exceptions.NotEnoughArgs(1, 0, "Memory")
        elif category is not None and prompt is None:
            pass
        elif category is None and prompt is not None:
            pass
        else:
            pass
        
    def _load_object_handler(self, path):
        loc = path + "/Data/object_dict.pickle"
        if os.paht.exists(loc):
            with open()
        
            
    def save(self):
        self.file_saver.save_model(self.model, "Memory/Data/memory_model.keras", 'memory model')