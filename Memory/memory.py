import numpy as np
import os
import pickle
import threading
import logging
from Memory.object_handler import Handler as Oh
from Memory.memory_model import MemModel
from Memory.category import Category

class Memory:
    def __init__(self, exceptions, saver, logger = None, path="Memory/Data", should_log = True, module_loader = None, file_loader = None):
        # Setting up utilities
        # Loading save files
        if file_loader is not None:
            self.file_loader = file_loader
        # Default Data Path
        self.path = path
        # Exceptions
        if exceptions is not None:
            self.exceptions = exceptions
        # Logging
        if logger is not None:
            self.logger = logger
            self.should_log = should_log
        # Module for loading classes
        if module_loader is not None:
            self.module_loader = module_loader
            self.model = module_loader.load(MemModel)
        self.file_saver = saver
        self._load_object_handler(path)
        model_file = f"{path}/Model/memory_model.keras"
        # Memory model
        if file_loader.exists(model_file):
            self.memory_arr = file_loader.load(path + "/Model/memory_model.keras")
        else:
            self._load_object_handler(path)
        # Loading the object handler
        if module_loader:
            self.object_handler = module_loader.load(Oh)
        else:
            self.object_handler = Oh()
        if self.should_log:
            self.logger.log(logging.INFO, 'Successfully loaded Memory')
                
    def get_dict(self, key):
        self.logger.log(logging.DEBUG, key)
        if key in self.object_dict.keys():
            return self.object_dict[key]
        else:
            self.logger.log(logging.INFO, f'Word {key} not found. Creating.')
            self._new_word(key)
            
    def Test(self):
        self.model.Test()
        
    def _new_word(self, word:list|str):
        if type(word) is list:
            for i in word:
                self.object_dict[i] = self.model.create_value(i)
        elif type(word) is str:
            self.object_dict[word] = self.model.create_value(word)
            
    def memory_call(self, category=None, prompt=None, *args, **kwargs):
        if category is None and prompt is None:
            raise self.exceptions.NotEnoughArgs(1, 0, "Memory")
        elif category is not None and prompt is None:
            if category in self.category_dict:
                pass
            else:
                return None
        elif category is None and prompt is not None:
            pass
        else:
            pass
        
    def _load_object_handler(self, path):
        loc = path + "/Data/object_dict.pickle"
        if self.file_loader.exists(loc):
            item = self.file_loader.load(loc)
            if item is None:
                self.object_dict = {}
            else:
                self.object_dict = item
            item1 = self.file_loader(loc)
            if item1 is None:
                self.category_dict = {}
            else:
                self.category_dict = item1
        else:
            self.logger.log(logging.INFO, 'No file was found with saved objects, creating a new one.')
            self.object_dict = {}
            self.category_dict = {}
            
    def save(self, saver):
        saver.save_other(self.object_dict, self.path+"/Data/object_dict.pickle")
        saver.save_model(self.model, self.path+"/Model/memory_model.keras", 'memory model')